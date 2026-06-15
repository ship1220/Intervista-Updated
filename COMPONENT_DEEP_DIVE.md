# Deep Dive: Key AI Components

This document details the 4 core AI/ML components that drive Intervista's functionality.

---

## 1. InterviewQuestionChain

**Location:** [core/chains/base_chain.py](core/chains/base_chain.py#L305)

**Purpose:** Generate role-aware, non-repeating interview questions tailored to candidate's resume and course progress.

### Inputs

```python
input_data = {
    "role": "Backend Engineer",              # Job role (e.g., "Software Engineer", "Data Scientist")
    "level": "Mid",                          # Experience level (Junior, Mid, Senior)
    "resume_text": "...",                    # Extracted resume content
    "completed_modules": ["Module A", ...],  # Courses the candidate completed
    "course_topics": ["Design Patterns", ...], # Topics from adaptive course
    "previous_questions": ["Q1", "Q2", ...], # Questions already asked (avoid repeats)
    "used_categories": ["technical", ...],   # Categories already covered (avoid repeats)
    "count": 5,                              # Number of questions to generate (ignored, generates 1 per call)
}
```

### Processing Flow

```
1. Prepare input data (convert lists to comma-separated strings)
2. Fetch from RAG: RetrievalContext
   └─ Query: "interview questions for {role} at {level} level"
   └─ Category: DocumentCategory.INTERVIEW_QUESTIONS
   └─ Returns: Similar questions + best practices from seed_data
3. Format prompt using prompt_manager.get_prompt("interviewer_system_prompt", **prepared)
4. Call LLMService.invoke(prompt, json_mode=True, use_cache=False)
5. Parse JSON response
6. Return ChainResult with JSON
```

### Output

```json
{
  "question": "Explain your approach to designing a scalable REST API...",
  "category": "technical"
}
```

**Status:** `success` or `error`

**Response Codes:**
- 0-100 difficulty level (implicit in category)
- `category` can be: `behavioral|situational|technical|logical|project-specific`

---

### LLM Prompt Template

**Prompt Name:** `interviewer_system_prompt` (v3.0)

**Located in:** [core/prompts/prompt_manager.py](core/prompts/prompt_manager.py#L110)

**Template:**

```
You are an agentic technical interviewer. Your job is to generate ONE interview question.

Context:
- Role: {role} at {level} level
- Resume highlights: {resume_text}
- Completed modules: {completed_modules}
- Course topics: {course_topics}
- Previous questions: {previous_questions}
- Used categories: {used_categories}

Rules:
1. Generate ONE unique question NOT in previous_questions
2. Prioritize weak areas from course_topics
3. Use resume_text to make questions relevant to candidate's background
4. Vary category (technical, behavioral, logical, situational, project-specific)
5. Avoid repeating used_categories
6. Use best practices from examples

Return ONLY JSON:
{
 "question": "...",
 "category": "behavioral|situational|technical|logical|project-specific"
}
```

**Key Variables:**
- `{role}`: Backend Engineer, Frontend Engineer, Data Scientist, etc.
- `{level}`: Junior, Mid, Senior
- `{resume_text}`: Extracted skills, projects, experience
- `{completed_modules}`: Course topics completed
- `{course_topics}`: Topics for this adaptive course
- `{previous_questions}`: Questions already asked (newline-separated)
- `{used_categories}`: Categories already used (comma-separated)

---

## 2. EvaluationChain

**Location:** [core/chains/base_chain.py](core/chains/base_chain.py#L350)

**Purpose:** Score interview answers on 5 dimensions with CKFS metrics for RL reward calculation.

### Inputs

```python
input_data = {
    "role": "Backend Engineer",              # Job role
    "level": "Mid",                          # Experience level
    "question": "Design a scalable REST API...", # Interview question asked
    "answer": "First, I would consider the...", # Candidate's answer (from speech transcription)
}
```

### Processing Flow

```
1. Prepare input data
2. Fetch from RAG: RetrievalContext
   └─ Query: "evaluate answer for {role}"
   └─ Category: DocumentCategory.EVALUATION_RUBRICS
   └─ Returns: Code quality rubric, communication rubric
3. Format prompt using prompt_manager.get_prompt("evaluate_answer", **prepared)
4. Call LLMService.invoke(prompt, json_mode=True, use_cache=False)
5. Parse JSON response
6. Extract 5 dimension scores for aggregation
7. Return ChainResult with evaluation JSON
```

### Output

```json
{
  "score": 75,
  "relevance_score": 80,
  "explanation_depth_score": 72,
  "star_method_score": 70,
  "structured_thinking_score": 78,
  "problem_solving_score": 75,
  "strengths": [
    "Clear API design principles",
    "Considered scalability"
  ],
  "weaknesses": [
    "Didn't mention caching strategy",
    "Limited discussion of failure handling"
  ],
  "ideal_answer": "I'd start by clarifying requirements: expected QPS, data volume, latency requirements...",
  "weak_topics": ["caching", "error handling"],
  "C": 0.75,
  "K": 0.72,
  "F": 0.70,
  "S": 0.78
}
```

**Output Fields:**
- **score**: 0-100 overall score
- **5 dimensions**: relevance, depth, STAR method, structured thinking, problem solving (each 0-100)
- **strengths/weaknesses**: Array of strings
- **ideal_answer**: First-person ideal response (used in report)
- **weak_topics**: List of skill gaps identified
- **CKFS metrics**: 
  - **C**: Conceptual correctness (0-1)
  - **K**: Knowledge depth (0-1)
  - **F**: Fluency/communication (0-1)
  - **S**: Structure/organization (0-1)

---

### LLM Prompt Template

**Prompt Name:** `evaluate_answer` (v4.0)

**Located in:** [core/prompts/prompt_manager.py](core/prompts/prompt_manager.py#L166)

**Template:**

```
You are evaluating an interview answer for a {level} {role} candidate.

IMPORTANT: The answer is from speech-to-text. Ignore spelling/grammar/STT mistakes.
Judge intent and substance. Do not penalize homophones or minor word errors.

Scoring rules:
- 0 → skipped or empty
- 10-30 → wrong or no substance
- 50-70 → partially correct, weak structure
- 80-90 → solid, relevant, reasonably structured
- 90-100 → excellent depth and structure

Dimension scoring (each 0-100, must reflect THIS answer):
- relevance_score: Does the answer address the question asked?
- explanation_depth_score: Technical/role depth and examples
- star_method_score: Behavioral answers — Situation, Task, Action, Result (0 if not behavioral)
- structured_thinking_score: Logical flow, steps, cause-effect, signposting
- problem_solving_score: Approach, trade-offs, solution quality (0 if not problem-solving)

Question:
{question}

Candidate Answer:
{answer}

Generate an ideal candidate response in first-person. The ideal answer should be a polished, 
interview-ready reply with concrete examples, clear structure, and personal ownership.

Return ONLY valid JSON:
{
 "score": <number 0-100>,
 "relevance_score": <number 0-100>,
 "explanation_depth_score": <number 0-100>,
 "star_method_score": <number 0-100>,
 "structured_thinking_score": <number 0-100>,
 "problem_solving_score": <number 0-100>,
 "strengths": ["..."],
 "weaknesses": ["..."],
 "ideal_answer": "A concise first-person candidate answer with example...",
 "weak_topics": ["topic1", "topic2"],
 "C": <float 0.0-1.0>,
 "K": <float 0.0-1.0>,
 "F": <float 0.0-1.0>,
 "S": <float 0.0-1.0>
}
```

**Key Variables:**
- `{role}`: Backend Engineer, Frontend Engineer, etc.
- `{level}`: Junior, Mid, Senior
- `{question}`: The interview question asked
- `{answer}`: Transcribed candidate answer from Whisper

---

## 3. create_course_internal()

**Location:** [main.py](main.py#L3055)

**Purpose:** Generate adaptive course skeleton + module structure based on RL bandit action and weak topics.

### Function Signature

```python
async def create_course_internal(
    user,                        # User object (models.User)
    role: str,                   # Job role
    level: str,                  # Experience level (Junior/Mid/Senior)
    weak_topics: list[str],      # Weak areas from interview evaluation
    action: str,                 # Bandit action (revision|easy|mixed|advanced)
    db: Session,                 # SQLAlchemy DB session
    topics: list[str] = None,    # Explicit topics to cover (if None, uses weak_topics)
    difficulty: str = "medium",  # Course difficulty (easy|medium|hard)
) -> int | None:
```

### Inputs

```python
# Example call from api_interview_evaluate():
new_course_id = await create_course_internal(
    user=user,
    role="Backend Engineer",
    level="Mid",
    weak_topics=["Design Patterns", "Distributed Systems"],
    action="mixed",               # From RL bandit selection
    db=db,
    topics=["Design Patterns", "Caching", "Load Balancing"],  # Normalized by _compose_course_topics()
    difficulty="intermediate"     # Mapped from action: mixed→intermediate
)
```

### Processing Flow

```
1. Normalize weak_topics (strip, deduplicate)
2. Set course_topics = topics if provided, else weak_topics
3. Map action to difficulty level:
   "revision" → "beginner"
   "easy" → "beginner"
   "mixed" → "intermediate"
   "advanced" → "hard"
4. Call _normalize_course_topics(role, topics, difficulty, max_topics=6)
5. Generate course title using _make_adaptive_course_title(...)
6. Build outline_prompt_input with strict_requirements from _course_generation_requirements(action, difficulty, topics)
7. Fetch prompt: prompt_manager.get_prompt("course_outline", **outline_prompt_input)
8. Call LLMService.invoke(prompt, json_mode=True)
   └─ Generates course outline with modules
9. Create Course object in DB:
   ├─ user_id, role, title, description
   ├─ level (difficulty)
   └─ status = "generated"
10. Create Module skeletons (3-5 modules):
    ├─ For each module:
    │  ├─ title, description (from LLM outline)
    │  ├─ order_index (0, 1, 2, ...)
    │  ├─ is_unlocked = True if first, else False (progressive unlock)
    │  ├─ is_final = True if last module
    │  └─ content = None (lazy-loaded on access)
    └─ db.add(module)
11. db.commit()
12. Return course.id
```

### Output

```python
course_id: int = 123  # Primary key of created Course record in DB

# Full Course object structure created:
Course(
    id=123,
    user_id=1,
    role="Backend Engineer",
    title="Advanced Backend Design Patterns",
    description="Master distributed systems, caching, and scalable architecture",
    level="intermediate",
    status="generated",
    created_at=datetime.now(),
    updated_at=datetime.now()
)

# With Module skeletons:
[
    Module(
        id=1,
        course_id=123,
        title="Design Patterns Fundamentals",
        description="Factory, Singleton, Observer",
        order_index=0,
        is_unlocked=True,      # First module unlocked
        is_final=False,
        content=None,          # Lazy-loaded later
        created_at=datetime.now()
    ),
    Module(
        id=2,
        course_id=123,
        title="Distributed Systems",
        description="CAP theorem, consensus, replication",
        order_index=1,
        is_unlocked=False,     # Locked until Module 1 complete
        is_final=False,
        content=None,
        created_at=datetime.now()
    ),
    # ... more modules
]
```

---

### LLM Prompt Template

**Prompt Name:** `course_outline` (v2.0)

**Located in:** [core/prompts/prompt_manager.py](core/prompts/prompt_manager.py#L200)

**Template:**

```
You are a senior curriculum designer.

Role/skill: {skill}
Learner designation: {level}
Duration: {duration_hours} hours
REQUIRED curriculum difficulty: {target_difficulty}
Bandit learning path: {bandit_action}
Suggested course title (use exactly or very close): {title_hint}

{strict_requirements}

Rules:
- Module depth MUST match REQUIRED curriculum difficulty 
  (easy=foundational, hard=advanced/expert).
- Do NOT produce a beginner course when difficulty is hard/advanced.
- Do NOT produce an expert course when difficulty is easy.
- First 2 modules must cover the mandated weak topics.
- course_title must reflect difficulty and topics (not a generic name).

Return JSON:
{
 "course_title": "Advanced Design Patterns & Distributed Systems",
 "description": "Master scalable architecture...",
 "learning_objectives": ["Understand distributed consensus", "..."],
 "modules": [
  {
   "module_title": "Design Patterns Fundamentals",
   "duration_minutes": 60,
   "topics": ["Factory", "Singleton", "Observer"]
  },
  {
   "module_title": "Distributed Systems",
   "duration_minutes": 90,
   "topics": ["CAP theorem", "Consensus algorithms"]
  }
 ],
 "assessments": ["Quiz after module 1", "..."]
}
```

**Key Variables:**
- `{skill}`: Backend Engineer, Frontend Engineer, Data Scientist
- `{level}`: Junior, Mid, Senior
- `{duration_hours}`: 20 (fixed)
- `{target_difficulty}`: "Foundational (for beginners)", "Intermediate", "Advanced (for experts)"
- `{bandit_action}`: "revision", "easy", "mixed", "advanced"
- `{title_hint}`: Pre-generated title hint (e.g., "Advanced Design Patterns")
- `{strict_requirements}`: Action-specific mapping (e.g., "Focus on revision of basics" or "Deep dive into advanced topics")

---

## 4. ContextualBandit

**Location:** [services/rl/rl_service.py](services/rl/rl_service.py#L60)

**Purpose:** Multi-armed bandit for adaptive course selection based on interview performance and learning history.

### Class Definition

```python
class ContextualBandit:
    """
    Contextual Multi-Armed Bandit agent for adaptive learning path selection.
    
    Unlike Q-learning:
    - No temporal difference or Bellman updates
    - No next_state dependency
    - Simple running average of rewards per (state, action) pair
    - Faster convergence for immediate feedback
    """
    
    def __init__(self, db: Session, action_space: str = "course"):
        self.db = db
        self.action_space = action_space
        # action_space: "interview" (5 actions) or "course" (4 actions)
```

### Key Methods

#### 1. `select_action()`

**Purpose:** Select which course difficulty/type to recommend next using ε-greedy + softmax

**Signature:**

```python
def select_action(
    state_id: str,                    # e.g., "low-2", "medium-3", "high-0"
    user_state: Optional[UserState],  # User RL state tracking
    last_action: Optional[str],       # Previously recommended action
    consecutive_action_count: int,    # How many times last_action was used
) -> str:  # Returns action name: "revision", "easy", "mixed", "advanced"
```

**Algorithm:**

```
Input: state_id="medium-3", user_state (session_count=5), last_action="easy", consecutive_count=2

1. Extract session_count from user_state
   └─ session_count = 5

2. Compute exploration rate
   └─ epsilon = 1.0 / (1.0 + session_count) = 1.0 / 6.0 ≈ 0.167

3. Get allowed actions for this state
   └─ If state_id = "medium-3":
      └─ allowed_actions = ["easy", "mixed"] (from ACTION_STATE_CONSTRAINTS)

4. Fetch Q-values from QTable
   └─ q_values = {
        "easy": 0.35,
        "mixed": 0.42
      }

5. COLD-START check
   └─ IF session_count < 2:
      └─ Return: "easy" (cold-start action)

6. EXPLORATION vs EXPLOITATION (ε-greedy)
   ├─ Roll: random_value = 0.08
   ├─ IF random_value < epsilon (0.08 < 0.167):
   │  └─ MODE: EXPLORE
   │  └─ Action: random choice from allowed_actions
   │  └─ Selected: "easy"
   └─ ELSE:
      └─ MODE: EXPLOIT
      └─ Action: argmax(adjusted_q_values)
      └─ Apply repetition penalty: penalty = 0.05 × 2 = 0.10
      └─ adjusted_q_values = {
           "easy": 0.35 - 0.10 = 0.25,
           "mixed": 0.42 (no penalty)
         }
      └─ Selected: "mixed"

7. SOFTMAX selection (if exploiting)
   ├─ Softmax temperature: 0.35
   ├─ exp_values = [exp((0.25-0.42)/0.35), exp((0.42-0.42)/0.35)]
   ├─ Normalize probabilities
   └─ Sample from distribution

8. Return selected action
```

**Inputs:**
- `state_id` (str): Discretized state from RL helpers (format: "{score_level}-{weak_count}")
  - score_level: "low" (<50), "medium" (50-75), "high" (>75)
  - weak_count: number of weak topics (0, 1, 2, 3+)
  - Example: "low-2" = score < 50 with 2 weak topics
- `user_state` (UserState): Contains session_count for epsilon calculation
- `last_action` (str): Previous action ("revision", "easy", "mixed", "advanced")
- `consecutive_action_count` (int): How many times last_action was recommended consecutively

**Output:**
```python
action: str = "mixed"  # One of: "revision", "easy", "mixed", "advanced"
```

**Action Meanings (for course context):**
- **"revision"**: Beginner difficulty, focus on fundamentals (soft reset)
- **"easy"**: Beginner difficulty, focus on weak areas (gentle)
- **"mixed"**: Intermediate difficulty, blend weak + general topics
- **"advanced"**: Advanced difficulty, deep dive on weak areas only

---

#### 2. `update_action_value()`

**Purpose:** Update Q-value (running average reward) for (state, action) pair

**Signature:**

```python
def update_action_value(
    state_id: str,      # e.g., "low-2"
    action_id: str,     # e.g., "easy"
    reward: float,      # Computed reward (-1.0 to +1.0)
) -> Tuple[float, float]:  # Returns (old_q, new_q)
```

**Algorithm (Running Average):**

```
Input: state_id="low-2", action_id="easy", reward=0.35

1. Query QTable for existing record
   ├─ SELECT * FROM q_table WHERE state_id="low-2" AND action_id="easy"
   └─ Result: existing Q-record with q_value=0.28, visit_count=3

2. IF record exists:
   ├─ old_q = 0.28
   ├─ visit_count = 3
   ├─ Formula: new_q = (old_q × visit_count + reward) / (visit_count + 1)
   ├─ new_q = (0.28 × 3 + 0.35) / 4 = (0.84 + 0.35) / 4 = 0.298
   ├─ visit_count = 4
   └─ UPDATE q_table SET q_value=0.298, visit_count=4 WHERE ...
   
3. ELSE (first time):
   ├─ Create new record
   ├─ new_q = reward = 0.35
   ├─ visit_count = 1
   └─ INSERT INTO q_table VALUES (..., state_id="low-2", action_id="easy", 0.35, 1)

4. db.commit()

5. Return: (0.28, 0.298)  # (old_q, new_q)
```

**Inputs:**
- `state_id` (str): Discretized state (e.g., "low-2")
- `action_id` (str): Action taken (e.g., "easy")
- `reward` (float): Normalized reward signal [-1.0, +1.0]
  - Calculated as: `0.5×score_improvement + 0.3×weak_topic_progress + 0.2×confidence_improvement`

**Output:**
```python
old_q: float = 0.28  # Previous Q-value
new_q: float = 0.298 # Updated Q-value
```

**Side Effect:**
- Updates QTable in database with new Q-value and visit count
- Logs Q-value update via `log_q_value_update(...)`

---

### State Space

**Format:** `"{score_level}-{weak_count}"`

**Example State Transitions:**

```
Interview 1: score=45, weak_topics=["API Design", "Caching"]
  └─ state_id = "low-2"
  └─ allowed_actions = ["revision", "easy"]
  └─ bandit selects: "easy"
  └─ course generated: beginner difficulty on weak areas

Interview 2: score=62, weak_topics=["Caching"]
  └─ state_id = "medium-1"
  └─ allowed_actions = ["easy", "mixed"]
  └─ Q-values: easy=0.4, mixed=0.5
  └─ bandit selects: "mixed"
  └─ course generated: intermediate difficulty on mixed topics

Interview 3: score=78, weak_topics=[]
  └─ state_id = "high-0"
  └─ allowed_actions = ["mixed", "advanced"]
  └─ Q-values: mixed=0.6, advanced=0.7
  └─ bandit selects: "advanced"
  └─ course generated: advanced difficulty on deep topics
```

---

### Action-State Constraints

```python
ACTION_STATE_CONSTRAINTS = {
    "low": ["revision", "easy"],        # Low score: revisit basics or gentle weak areas
    "medium": ["easy", "mixed"],        # Medium score: weak areas or mixed progression
    "high": ["mixed", "advanced"],      # High score: mixed or deep advanced topics
}
```

---

### Hyperparameters

```python
PENALTY_LAMBDA = 0.05                      # Penalty per consecutive repetition
SOFTMAX_TEMPERATURE = 0.35                 # Softmax temperature for action selection
COLD_START_THRESHOLD = 2                   # Sessions before using learned policy
```

---

### Reward Calculation (from utils/rl_helpers.py)

```python
def calculate_reward(
    current_score: float,           # New interview score (0-100)
    previous_score: float,          # Last interview score
    current_weak_topics: list[str], # Current weak areas
    previous_weak_topics: list[str],# Previous weak areas
    current_confidence: float,      # Current confidence score
    previous_confidence: float,     # Previous confidence score
) -> float:  # Returns [-1.0, +1.0]
```

**Calculation:**

```python
# Component 1: Score improvement
score_improvement = (current_score - previous_score) / 100.0  # Normalize to [-1, +1]

# Component 2: Weak topic progress
if previous_weak_topics:
    prev_set = set(t.lower().strip() for t in previous_weak_topics)
    curr_set = set(t.lower().strip() for t in current_weak_topics)
    overlap = len(prev_set.intersection(curr_set))
    weak_topic_progress = max(0.0, min(1.0, (len(prev_set) - overlap) / 3.0))
else:
    weak_topic_progress = 0.0

# Component 3: Confidence improvement
if previous_confidence is not None:
    confidence_improvement = (current_confidence - previous_confidence) / 100.0
else:
    confidence_improvement = 0.0

# Weighted combination
reward = (
    0.5 * score_improvement +
    0.3 * weak_topic_progress +
    0.2 * confidence_improvement
)

# Clamp to [-1.0, +1.0]
reward = max(-1.0, min(1.0, reward))
```

**Example:**

```
previous_score = 45, current_score = 62
  └─ score_improvement = (62 - 45) / 100 = 0.17

previous_weak_topics = ["API Design", "Caching", "Load Balancing"]
current_weak_topics = ["Caching"]
  └─ Improvement: 2 topics resolved out of 3
  └─ weak_topic_progress = (3 - 1) / 3 = 0.67

previous_confidence = 40, current_confidence = 65
  └─ confidence_improvement = (65 - 40) / 100 = 0.25

reward = 0.5×0.17 + 0.3×0.67 + 0.2×0.25
       = 0.085 + 0.201 + 0.05
       = 0.336
```

---

## Summary Comparison

| Component | Input Type | Output Type | LLM | RL |
|-----------|-----------|-------------|-----|-----|
| **InterviewQuestionChain** | Dict with role, resume, history | JSON: {question, category} | Groq (Llama 4) | No |
| **EvaluationChain** | Dict with Q&A | JSON: {scores, weaknesses, ideal_answer} | Groq (Llama 4) | No (but output used by RL) |
| **create_course_internal()** | Course params + action | Course object + Modules in DB | Groq (Llama 4, course_outline prompt) | Yes (action from bandit) |
| **ContextualBandit** | State + history | Action string | No | Yes (core algorithm) |

---

## Complete Flow Example

```
1. Interview evaluation completes
   └─ score=62, weak_topics=["Caching", "Load Balancing"]
   
2. RL Bandit:
   ├─ Discretize: state_id = get_state_id(62, 2) = "medium-2"
   ├─ Get allowed actions: ["easy", "mixed"]
   ├─ Query QTable: easy=0.4, mixed=0.45
   ├─ ε-greedy: select "mixed"
   └─ Action: "mixed"
   
3. Course Creation:
   ├─ Map action: "mixed" → difficulty="intermediate"
   ├─ Call create_course_internal(..., action="mixed", difficulty="intermediate")
   ├─ LLM generates outline with 5 modules
   ├─ Create Course + Module skeletons
   └─ Return course_id=123
   
4. User views course
   ├─ Click Module 2: "Load Balancing Basics"
   ├─ API: GET /api/course/123/module/2/content
   ├─ LLM generates: markdown content (lazy-loaded)
   ├─ Return content + cache in DB
   └─ User reads module
   
5. After course completion
   ├─ calculate_reward(...) = 0.35 (improvement signal)
   ├─ bandit.update_action_value("medium-2", "mixed", 0.35)
   ├─ new_q = (0.45 × 3 + 0.35) / 4 = 0.44
   ├─ Update QTable: mixed Q-value now 0.44
   └─ Ready for next interview cycle
```
