"""
Structured terminal logging for contextual bandit / course generation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _line(char: str = "=", width: int = 56) -> str:
    return char * width


def log_section(title: str) -> None:
    print(f"\n{_line()}", flush=True)
    print(f"  {title}", flush=True)
    print(_line(), flush=True)


def log_kv(label: str, value: Any, indent: int = 2) -> None:
    prefix = " " * indent
    print(f"{prefix}{label}: {value}", flush=True)


def log_bandit_state(
    *,
    state_id: str,
    overall_score: float,
    weak_topics: list,
    weak_count: int,
    previous_score: float,
    prev_state_id: str,
    session_count: int,
) -> None:
    log_section("BANDIT STATE DISCRETIZATION")
    log_kv("Overall Score", f"{overall_score:.1f}")
    log_kv("Previous Avg Score", f"{previous_score:.1f}")
    log_kv("Weak Topics Detected", weak_topics or ["(none)"])
    log_kv("Weak Topic Count", weak_count)
    log_kv("Current State ID", state_id)
    log_kv("Previous State ID", prev_state_id)
    log_kv("Session Count", session_count)


def log_bandit_action_selection(
    *,
    state_id: str,
    q_values: Dict[str, float],
    epsilon: float,
    session_count: int,
    mode: str,
    selected_action: str,
) -> None:
    log_section("BANDIT ACTION SELECTION")
    log_kv("Current State", state_id)
    log_kv("Exploration Rate (epsilon)", f"{epsilon:.0%}")
    log_kv("Session Count", session_count)
    log_kv("Selection Mode", mode)
    print("  Q-values:", flush=True)
    if not q_values:
        print("    (no Q-values stored yet — cold start / unexplored)", flush=True)
    else:
        for action, q in sorted(q_values.items(), key=lambda x: -x[1]):
            bar = "#" * max(1, int(abs(q) * 20)) if q else ""
            print(f"    {action:12} {q:+.4f}  {bar}", flush=True)
    log_kv("Selected Action", selected_action, indent=2)


def log_reward_calculation(
    *,
    overall_score: float,
    previous_score: float,
    score_improvement: float,
    reward: float,
    weak_topic_progress: float | None = None,
    confidence_improvement: float | None = None,
) -> None:
    log_section("REWARD CALCULATION")
    log_kv("Current Score", f"{overall_score:.1f}")
    log_kv("Previous Score", f"{previous_score:.1f}")
    log_kv("Score Improvement", f"{score_improvement:+.1f}")
    if weak_topic_progress is not None:
        log_kv("Weak Topic Progress", f"{weak_topic_progress:+.3f}")
    if confidence_improvement is not None:
        log_kv("Confidence Improvement", f"{confidence_improvement:+.3f}")
    log_kv("Normalized Reward", f"{reward:+.4f}  (clamped to [-1, +1])")


def log_q_value_update(
    *,
    state_id: str,
    action_id: str,
    reward: float,
    old_q: float,
    new_q: float,
    visit_count: int,
) -> None:
    log_section("Q-VALUE LEARNING UPDATE")
    log_kv("State", state_id)
    log_kv("Action", action_id)
    log_kv("Reward Applied", f"{reward:+.4f}")
    log_kv("Q-value Before", f"{old_q:+.4f}")
    log_kv("Q-value After", f"{new_q:+.4f}")
    log_kv("Visit Count", visit_count)
    delta = new_q - old_q
    log_kv("Delta", f"{delta:+.4f}")


def log_course_generation_decision(
    *,
    action: str,
    course_topics: list,
    course_difficulty: str,
    weak_topics: list,
    new_course_id: Optional[int],
    course_title: Optional[str] = None,
    fallback_used: bool = False,
) -> None:
    log_section("ADAPTIVE COURSE GENERATION")
    log_kv("Bandit Action", action)
    log_kv("Mapped Difficulty", course_difficulty)
    log_kv("Weak Topics (from interview)", weak_topics or ["(none)"])
    log_kv("Selected Course Topics", course_topics)
    if fallback_used:
        log_kv("Note", "Fallback course generation was used")
    if new_course_id:
        log_kv("Generated Course ID", new_course_id)
        if course_title:
            log_kv("Course Title", course_title)
    else:
        log_kv("Generated Course ID", "FAILED — no course created")


def log_bandit_complete(
    *,
    state_id: str,
    action: str,
    reward: float,
    new_course_id: Optional[int],
) -> None:
    log_section("BANDIT PIPELINE COMPLETE")
    log_kv("Final State", state_id)
    log_kv("Final Action", action)
    log_kv("Final Reward", f"{reward:+.4f}")
    log_kv("Course ID", new_course_id or "none")
    print(_line(), flush=True)
