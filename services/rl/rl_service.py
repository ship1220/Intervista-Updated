"""
Reinforcement Learning Service
================================

Implements Contextual Multi-Armed Bandit for adaptive interview and course selection.

Key Features:
- Contextual bandits (state-aware action selection)
- ε-greedy action selection with cold-start handling
- Running average reward tracking (no temporal difference)
- State-action reward management
- Production-safe logging

Usage:
    bandit = ContextualBandit(db_session)
    action = bandit.select_action(state_id, user_state)
    bandit.update_action_value(state_id, action_id, reward)
"""

import logging
import math
import random
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from models import QTable, UserState
from utils.bandit_logger import log_bandit_action_selection

logger = logging.getLogger(__name__)

# ============================================================================
# ACTION SPACE DEFINITIONS
# ============================================================================

INTERVIEW_ACTIONS = {
    "ask_easy_question": 0,
    "ask_medium_question": 1,
    "ask_hard_question": 2,
    "ask_resume_question": 3,
    "ask_behavioral_question": 4,
}

COURSE_ACTIONS = {
    "revision": 0,
    "easy": 1,
    "mixed": 2,
    "advanced": 3,
}

# Action constraints by score band for course recommendations
ACTION_STATE_CONSTRAINTS = {
    "low": ["revision", "easy"],
    "medium": ["easy", "mixed"],
    "high": ["mixed", "advanced"],
}

# Hyperparameters
PENALTY_LAMBDA = 0.05  # Penalty per consecutive repetition
SOFTMAX_TEMPERATURE = 0.35
COLD_START_THRESHOLD = 2  # Sessions before using learned policy


class ContextualBandit:
    """
    Contextual Multi-Armed Bandit agent for adaptive interview and course selection.

    This bandit maintains action-reward associations per state and uses
    ε-greedy exploration to balance exploration vs exploitation.
    
    Key difference from Q-learning:
    - No temporal difference or Bellman updates
    - No next_state dependency
    - Simple running average of rewards per (state, action) pair
    - Faster convergence for immediate reward feedback
    """

    def __init__(self, db: Session, action_space: str = "interview"):
        """
        Initialize the bandit.

        Args:
            db: SQLAlchemy session for database access
            action_space: Either 'interview' or 'course'
        """
        self.db = db
        self.action_space = action_space

        if action_space == "interview":
            self.actions = INTERVIEW_ACTIONS
        elif action_space == "course":
            self.actions = COURSE_ACTIONS
        else:
            raise ValueError(f"Unknown action space: {action_space}")

    # ========================================================================
    # ACTION SELECTION (ε-GREEDY)
    # ========================================================================

    def select_action(
        self,
        state_id: str,
        user_state: Optional[UserState],
        last_action: Optional[str] = None,
        consecutive_action_count: int = 0,
    ) -> str:
        """
        Select an action using adaptive exploration and softmax selection.

        - Cold-start: first sessions use a safe easy/revision path.
        - Exploration rate ε(t) = 1 / (1 + t).
        - Softmax selection over adjusted Q-values prevents premature convergence.
        - State constraints enforce age-appropriate action sets.
        - Repetition penalty discourages action loops.

        Args:
            state_id: Current discretized state
            user_state: User state record (optional, for session count)
            last_action: Most recently recommended action
            consecutive_action_count: Number of consecutive times the last_action was used

        Returns:
            Selected action name (string)
        """
        session_count = user_state.session_count if user_state else 0
        epsilon = 1.0 / (1.0 + session_count)
        allowed_actions = self._get_allowed_actions(state_id)
        q_values = self.get_q_value_dict(state_id)
        adjusted_q_values = self._adjust_q_values(
            q_values, last_action, consecutive_action_count, allowed_actions
        )

        # ====== COLD START ======
        if session_count < COLD_START_THRESHOLD:
            action = self._cold_start_action()
            mode = f"COLD START (sessions < {COLD_START_THRESHOLD})"
            log_bandit_action_selection(
                state_id=state_id,
                q_values=q_values,
                epsilon=epsilon,
                session_count=session_count,
                mode=mode,
                selected_action=action,
            )
            return action

        explore_roll = random.random()
        if explore_roll < epsilon:
            action = random.choice(allowed_actions)
            mode = f"EXPLORE (roll={explore_roll:.3f} < ε={epsilon:.3f})"
        else:
            action = self._softmax_action(adjusted_q_values, allowed_actions)
            mode = f"SOFTMAX (roll={explore_roll:.3f} >= ε={epsilon:.3f})"

        log_bandit_action_selection(
            state_id=state_id,
            q_values=q_values,
            epsilon=epsilon,
            session_count=session_count,
            mode=mode,
            selected_action=action,
        )
        return action

    def _cold_start_action(self) -> str:
        """Return the cold-start action for this action space."""
        if self.action_space == "interview":
            return "ask_easy_question"
        else:
            return "easy"

    def _get_allowed_actions(self, state_id: str) -> list[str]:
        """Return actions permitted by the user's current score band."""
        if self.action_space != "course":
            return list(self.actions.keys())

        score_level = str(state_id).split("-")[0] if state_id else "medium"
        allowed = ACTION_STATE_CONSTRAINTS.get(score_level, list(self.actions.keys()))
        if not allowed:
            return list(self.actions.keys())
        return allowed

    def _adjust_q_values(
        self,
        q_values: dict,
        last_action: Optional[str],
        consecutive_action_count: int,
        allowed_actions: list[str],
    ) -> dict:
        """Apply a small penalty for repeated consecutive recommendations."""
        adjusted = {}
        for action_name in allowed_actions:
            q = q_values.get(action_name, 0.0)
            if last_action and action_name == last_action and consecutive_action_count > 0:
                q -= PENALTY_LAMBDA * consecutive_action_count
            adjusted[action_name] = q
        return adjusted

    def _softmax_action(self, adjusted_q_values: dict, allowed_actions: list[str]) -> str:
        """Select an action by sampling from a softmax distribution over adjusted Q-values."""
        if not adjusted_q_values:
            return random.choice(allowed_actions)

        values = [adjusted_q_values.get(action, 0.0) for action in allowed_actions]
        if all(v == 0.0 for v in values):
            return random.choice(allowed_actions)

        max_val = max(values)
        exp_values = [math.exp((v - max_val) / SOFTMAX_TEMPERATURE) for v in values]
        total = sum(exp_values)
        if total == 0:
            return random.choice(allowed_actions)

        probabilities = [v / total for v in exp_values]
        return random.choices(allowed_actions, weights=probabilities, k=1)[0]

    def _greedy_action(self, state_id: str, q_values: dict = None) -> str:
        """
        Select the action with the highest Q-value (avg reward) for a given state.

        If no Q-values exist for the state, return a random action.

        Args:
            state_id: Current state ID
            q_values: Pre-computed Q-values dict (optional, for efficiency)

        Returns:
            Action with highest avg reward
        """
        if q_values is None:
            q_values = self.get_q_value_dict(state_id)
        
        if not q_values or all(v == 0.0 for v in q_values.values()):
            # No learned values yet, random action
            best_action = random.choice(list(self.actions.keys()))
            logger.debug(f"[BANDIT] GREEDY: No Q-values for state_id={state_id}, returning random action")
            return best_action

        best_action = max(q_values.keys(), key=lambda a: q_values.get(a, 0.0))
        return best_action

    # ========================================================================
    # Q-TABLE OPERATIONS (Q-VALUE = RUNNING AVERAGE OF REWARDS)
    # ========================================================================

    def _get_q_value(self, state_id: str, action_name: str) -> float:
        """
        Fetch Q-value (running average reward) for (state, action) pair.

        If not found in database, returns 0.0.

        Args:
            state_id: State identifier
            action_name: Action name

        Returns:
            Q-value (running average reward), or 0.0 if not found
        """
        try:
            q_record = (
                self.db.query(QTable)
                .filter(
                    QTable.state_id == state_id,
                    QTable.action_id == action_name,
                )
                .first()
            )

            if q_record:
                return float(q_record.q_value)
            else:
                return 0.0
        except Exception as e:
            logger.error(f"[BANDIT] Error fetching Q-value: {e}")
            return 0.0

    # ========================================================================
    # ACTION VALUE UPDATE (MAIN LEARNING STEP - RUNNING AVERAGE)
    # ========================================================================

    def update_action_value(
        self,
        state_id: str,
        action_id: str,
        reward: float,
    ) -> Tuple[float, float]:
        """
        Update action value using running average (no temporal difference).

        Formula:
            new_q = (old_q * visit_count + reward) / (visit_count + 1)

        This is a simple bandit update: just average the rewards received
        for this (state, action) pair. No next_state dependency.

        Args:
            state_id: Current state
            action_id: Selected action
            reward: Received reward (normalized 0-1)

        Returns:
            Tuple of (old_q_value, new_q_value)
        """
        try:
            # Fetch or create Q-record
            q_record = (
                self.db.query(QTable)
                .filter(
                    QTable.state_id == state_id,
                    QTable.action_id == action_id,
                )
                .first()
            )

            if q_record is None:
                q_record = QTable(
                    state_id=state_id,
                    action_id=action_id,
                    q_value=reward,  # First reward is the initial Q-value
                    visit_count=1,
                )
                self.db.add(q_record)
                self.db.flush()
                old_q = 0.0
                new_q = reward
            else:
                old_q = float(q_record.q_value)
                
                # Running average: new_q = (old_q * count + reward) / (count + 1)
                new_q = (old_q * q_record.visit_count + reward) / (q_record.visit_count + 1)
                
                q_record.q_value = new_q
                q_record.visit_count += 1

            self.db.commit()

            from utils.bandit_logger import log_q_value_update

            log_q_value_update(
                state_id=state_id,
                action_id=action_id,
                reward=reward,
                old_q=old_q,
                new_q=new_q,
                visit_count=q_record.visit_count,
            )

            return old_q, new_q

        except Exception as e:
            self.db.rollback()
            logger.error(f"[BANDIT] Error updating action value: {e}", exc_info=True)
            return 0.0, 0.0

    def update_q_table(
        self,
        state_id: str,
        action_id: str,
        reward: float,
        next_state_id: str = None,
    ) -> Tuple[float, float]:
        """
        Backward compatibility wrapper for update_action_value.

        This method exists for legacy code that may pass next_state_id.
        The next_state_id parameter is IGNORED (bandit doesn't use it).

        Args:
            state_id: Current state
            action_id: Selected action
            reward: Received reward
            next_state_id: IGNORED (for backward compatibility only)

        Returns:
            Tuple of (old_q_value, new_q_value)
        """
        logger.debug(
            f"[BANDIT] update_q_table (legacy API) called: ignoring next_state_id={next_state_id}"
        )
        return self.update_action_value(state_id, action_id, reward)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_q_value_dict(self, state_id: str) -> dict:
        """
        Get all Q-values (running average rewards) for a given state.

        Useful for logging and debugging.

        Args:
            state_id: State identifier

        Returns:
            Dict mapping action names to Q-values
        """
        try:
            q_dict = {}
            for action_name in self.actions.keys():
                q_dict[action_name] = self._get_q_value(state_id, action_name)
            return q_dict
        except Exception as e:
            logger.error(f"[BANDIT] Error fetching Q-value dict: {e}")
            return {}

    def reset_q_table(self) -> None:
        """
        DANGER: Reset all Q-values to 0.

        Use only for testing or policy reset. NOT recommended in production.
        """
        try:
            self.db.query(QTable).delete()
            self.db.commit()
            logger.warning("[BANDIT] Q-TABLE RESET: All Q-values deleted")
        except Exception as e:
            self.db.rollback()
            logger.error(f"[BANDIT] Error resetting Q-table: {e}")


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# For legacy code that imports TabularQLearner
TabularQLearner = ContextualBandit
