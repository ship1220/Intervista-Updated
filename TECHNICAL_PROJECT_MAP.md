# TECHNICAL PROJECT MAP - Intervista AI Interview System

---

## 1. COMPLETE FOLDER STRUCTURE & PURPOSES

```
Intervista-Updated/
│
├── main.py                          # FastAPI app entry point (all endpoints)
├── models.py                        # SQLAlchemy ORM models (10 tables)
├── database.py                      # SQLAlchemy session & engine setup
├── config/
│   └── settings.py                  # Environment config (Groq API, LLM params, RAG settings)
│
├── core/                            # AI Pipeline Components
│   ├── llm/
│   │   └── llm_service.py           # Groq API wrapper + LLM caching (MD5 key, TTL)
│   ├── chains/
│   │   ├── base_chain.py            # Pipeline orchestration (Input→RAG→Format→LLM→Parse)
│   │   └── __init__.py
│   └── prompts/
│       └── prompt_manager.py        # Prompt templates (INTERVIEW, EVALUATION, COURSE, etc.)
│
├── services/
│   ├── rag/                         # Retrieval-Augmented Generation
│   │   ├── rag_pipeline.py          # Orchestrator (retrieve + ingest + cache)
│   │   ├── retriever.py             # FAISS search interface
│   │   ├── vector_store.py          # FAISS backend + text embeddings (all-MiniLM-L6-v2)
│   │   ├── document_ingester.py     # Batch document validation & ingestion
│   │   ├── rag_config.py            # Document categories enum
│   │   └── seed_data.py             # Knowledge base (interview Qs, rubrics, courses, best practices)
│   │
│   └── rl/                          # Reinforcement Learning
│       └── rl_service.py            # Contextual bandit (action selection, Q-value updates)
│
├── evaluation/
│   └── evaluator.py                 # Chain testing & metric validation
│
├── speech/
│   └── transcription.py             # Whisper ASR + speech metrics (WPM, filler words, clarity)
│
├── utils/
│   ├── logger.py                    # Structured logging wrapper
│   ├── bandit_logger.py             # Terminal-friendly RL logging
│   ├── rl_helpers.py                # State discretization, reward calculation
│   └── logging_config.py            # Log level configuration
│
├── templates/                       # Jinja2 HTML templates
│   ├── index.html                   # Dashboard
│   ├── interview.html               # Voice interview UI
│   ├── report.html                  # Interview results viewer
│   ├── profile.html                 # User skill profile
│   ├── course.html                  # Course generation page
│   ├── module.html                  # Module/quiz viewer
│   ├── login.html, signup.html      # Auth pages
│   ├── home.html                    # Landing page
│   └── roadmap.html                 # Learning roadmap
│
├── static/                          # CSS stylesheets
│   ├── style.css, styles.css        # General styling
│   ├── profile-dashboard.css        # Profile layout
│   └── report.css                   # Report visualization
│
├── uploads/                         # Resume PDF/DOCX storage
│
└── Data files:
    ├── user_skill_profile.py        # Pydantic models for user skills (UserSkillProfile, UserSkillVector)
    ├── user_skill_integration.py    # Integration with profiles
    └── requirements.txt             # Dependencies (FastAPI, Groq, FAISS, Whisper, etc.)
```

---

## 2. MAIN FRONTEND PAGES & ROUTES

| Page | Route | File | Purpose |
|------|-------|------|---------|
| **Home** | `/` | `templates/home.html` | Landing page, project intro |
| **Login** | `/login` | `templates/login.html` | User authentication |
| **Signup** | `/signup` | `templates/signup.html` | User registration |
| **Dashboard** | `/index` | `templates/index.html` | Main hub (role/level selection, course list) |
| **Interview** | `/interview/start` | `templates/interview.html` | Voice interview (audio recording, transcription) |
| **Report** | `/report` | `templates/report.html` | Interview results (scores, feedback, ideal answers) |
| **Profile** | `/profile` | `templates/profile.html` | User skill profile, resume upload, history |
| **Course Gen** | `/generate_course/` | `templates/course.html` | Adaptive course creation |
| **Module View** | `/module` | `templates/module.html` | Course content + quiz |
| **Roadmap** | `/roadmap` | `templates/roadmap.html` | Personalized learning path |

---

## 3. MAIN FASTAPI ENDPOINTS

### **Authentication & User Management**
```python
POST   /login              # User login (cookie-based)
POST   /signup             # User registration
GET    /logout             # Clear session
```

### **Resume & Profile Processing**
```python
POST   /api/upload_resume  # Parse resume → Extract skills → LLM analysis → Store profile
POST   /update-resume      # Alternative resume upload endpoint
GET    /profile            # Fetch user skill profile
```

### **Interview Management**
```python
GET    /interview/start             # Initialize session (role, level, context)
POST   /start_interview/            # Alternative start endpoint
POST   /api/interview/next_question # Generate next Q (avoid repeats, consider weak areas)
POST   /api/interview/evaluate      # Evaluate all answers + generate 8-feature report
POST   /api/transcribe             # Whisper ASR (audio → text)
```

### **Course Management**
```python
POST   /api/course/generate                           # Bandit action selection → Generate course skeleton
GET    /api/course/{id}/module/{mid}/content         # Lazy-load module markdown content
GET    /api/course/{id}/module/{mid}/quiz            # Fetch quiz JSON
POST   /api/course/{id}/module/{mid}/quiz            # Submit quiz answers
GET    /course                                        # List user's courses
```

### **Profile & Reports**
```python
GET    /report             # View interview report
GET    /roadmap            # Get personalized learning roadmap
```

---

## 4. DATABASE MODELS/TABLES

| Table | Purpose | Key Fields |
|-------|---------|-----------|
| **users** | Auth credentials | `id, username, password (hashed)` |
| **user_profiles** | Extended profile | `user_id, role_applied_for, extracted_skills (JSON), skill_gaps, profile_json` |
| **interview_attempts** | Per-question records | `user_id, role, topic, answer, feedback, timestamp` |
| **interview_sessions** | Active interview state | `user_id, role, level, status, created_at` |
| **interviews** | High-level interview records | `user_id, role, date, score, report_json` |
| **skill_progress** | Weak skill tracking | `user_id, skill, attempts, weak (boolean)` |
| **user_skill_profile_row** | RL state tracking | `user_id, technical_skills (JSON), interview_skills (JSON), overall_score` |
| **user_state** | Bandit context | `user_id, state_id ("low-2"), avg_score, session_count` |
| **courses** | Course metadata | `user_id, role, title, level, status, created_at` |
| **modules** | Course sections | `course_id, title, content, quiz (JSON), is_completed` |
| **module_attempts** | Quiz attempts | `user_id, module_id, score, answers (JSON)` |
| **q_table** | RL Q-values | `state_id, action_id, q_value, visit_count` |

---

## 5. AI-RELATED FILES & COMPONENTS

### **RAG Pipeline**
- **[services/rag/rag_pipeline.py](services/rag/rag_pipeline.py)** - Orchestrates retrieval, ingestion, caching
- **[services/rag/retriever.py](services/rag/retriever.py)** - Semantic search (FAISS L2 distance, score 0-100)
- **[services/rag/vector_store.py](services/rag/vector_store.py)** - FAISS backend + Sentence-Transformers embeddings (384-dim)
- **[services/rag/seed_data.py](services/rag/seed_data.py)** - 7 knowledge base categories (interview Qs, rubrics, courses, best practices)
- **[services/rag/document_ingester.py](services/rag/document_ingester.py)** - Batch validation & ingestion

### **LLM Integration**
- **[core/llm/llm_service.py](core/llm/llm_service.py)** - Groq API wrapper (meta-llama/llama-4-scout-17b-16e-instruct)
  - Caching: MD5 hash of prompt, TTL 3600s, max 500 entries
  - JSON mode support for structured output
  - Async execution

### **LLM Chains**
- **[core/chains/base_chain.py](core/chains/base_chain.py)** - Pipeline (Input→RAG→Prompt→LLM→Parse)
  - `InterviewQuestionChain` - Generate role-aware questions
  - `EvaluationChain` - Score answers with 5 dimensions (relevance, depth, STAR, structure, problem-solving)
  - `SummaryChain` - Generate performance reports

### **Prompt Management**
- **[core/prompts/prompt_manager.py](core/prompts/prompt_manager.py)** - Centralized prompt templates
  - Categories: INTERVIEW, EVALUATION, COURSE, SUMMARY, PROFILE, RAG_RETRIEVAL

### **Speech Processing**
- **[speech/transcription.py](speech/transcription.py)** - Whisper ASR + metrics
  - Transcription, STT normalization, speech delivery analysis (WPM, filler words, clarity)

### **Reinforcement Learning**
- **[services/rl/rl_service.py](services/rl/rl_service.py)** - Contextual multi-armed bandit
  - Action spaces: INTERVIEW_ACTIONS (5), COURSE_ACTIONS (4)
  - State discretization: "{score_level}-{weak_count}" (e.g., "low-2", "high-0")
  - ε-greedy exploration (epsilon = 1/(1+session_count))
  - Running average Q-values
- **[utils/rl_helpers.py](utils/rl_helpers.py)** - State discretization, reward calculation

---

## 6. END-TO-END INTERVIEW GENERATION FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER UPLOADS RESUME                                          │
├─────────────────────────────────────────────────────────────────┤
│ POST /api/upload_resume                                         │
│   → Extract text (PyPDF/python-docx)                            │
│   → LLMService.invoke() with PROFILE prompt                     │
│   → Parse skills, gaps, strengths                               │
│   → Store in UserProfile + user_skill_profile.py               │
│   → session: resume_store[username] = resume_text              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. USER STARTS INTERVIEW                                        │
├─────────────────────────────────────────────────────────────────┤
│ GET /interview/start?role=Backend&level=Mid                     │
│   → Initialize session: interview_sessions[username]            │
│   → Load completed_modules, course_topics                       │
│   → Set context for question generation                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. GENERATE NEXT QUESTION (Per Question Loop)                   │
├─────────────────────────────────────────────────────────────────┤
│ POST /api/interview/next_question                               │
│   → Retrieve previous questions from history                    │
│   → Build LLM context:                                          │
│     - Resume skills, gaps, strengths                            │
│     - Completed courses/topics                                  │
│     - Used question categories (avoid repeats)                  │
│   → RAG.retrieve_context() if enabled                           │
│   → InterviewQuestionChain.invoke()                             │
│   → LLM returns: {"question": "...", "category": "..."}        │
│   → Store in memory (history)                                   │
│   → Return to frontend                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. USER RECORDS ANSWER                                          │
├─────────────────────────────────────────────────────────────────┤
│ Frontend captures audio (WebAudio API) → WAV file              │
│   → POST /api/transcribe (audio)                               │
│   → Whisper ASR: audio → text                                  │
│   → normalize_transcript() (STT error correction)               │
│   → Return normalized text to frontend                          │
│   → Store locally (session)                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. SUBMIT ALL ANSWERS FOR EVALUATION                            │
├─────────────────────────────────────────────────────────────────┤
│ POST /api/interview/evaluate                                    │
│   → Build interview context: (Q, A, role, resume) pairs        │
│   → EvaluationChain for each Q-A pair:                          │
│      - Score 0-100 with 5 dimensions                            │
│      - Relevance, depth, STAR, structure, problem-solving      │
│   → analyze_speech_delivery(answer, duration):                  │
│      - WPM (words per minute)                                   │
│      - Filler word count                                        │
│      - Clarity score, engagement score                          │
│   → Aggregate 8 features:                                       │
│      1. Candidate Profile (role, date, interview_count)         │
│      2. Speech Delivery (WPM, filler words, clarity)            │
│      3. Content Analysis (LLM scores)                           │
│      4. Confidence Score (aggregated)                           │
│      5. Performance Summary (LLM-generated)                     │
│      6. Overall Score & Recruiter Verdict                       │
│      7. Detailed Answer Report (per-question)                   │
│      8. RL-based Course Recommendation                          │
│   → Generate report JSON                                        │
│   → Store: interviews[Interview(user_id, role, score, report)]│
│   → Return complete report                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. VIEW REPORT                                                  │
├─────────────────────────────────────────────────────────────────┤
│ GET /report                                                     │
│   → Fetch interview from DB                                     │
│   → Render report.html with scores, feedback, ideal answers    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. END-TO-END COURSE GENERATION FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│ TRIGGER: Interview Evaluation Complete                          │
├─────────────────────────────────────────────────────────────────┤
│ POST /api/interview/evaluate → Generate report                  │
│   → Extract: overall_score, weak_topics_count                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1. RL BANDIT STATE DISCRETIZATION                               │
├─────────────────────────────────────────────────────────────────┤
│ utils/rl_helpers.py: get_state_id(avg_score, weak_count)       │
│   → Score bands:                                                │
│      - overall_score < 50 → "low"                               │
│      - 50-75 → "medium"                                         │
│      - > 75 → "high"                                            │
│   → state_id = f"{score_level}-{weak_count}"                   │
│   → Example: "low-2", "medium-3", "high-0"                     │
│   → Store in UserState (user_id, state_id)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. BANDIT ACTION SELECTION                                      │
├─────────────────────────────────────────────────────────────────┤
│ services/rl/rl_service.py: select_action(state_id, user_state)│
│                                                                 │
│   Cold-start: if session_count < 2                             │
│      → Return "revision" action (safe)                          │
│                                                                 │
│   ε-greedy:                                                     │
│      → epsilon = 1 / (1 + session_count)                        │
│      → If random() < epsilon:                                   │
│           - Explore: random action from valid set              │
│         Else:                                                   │
│           - Exploit: argmax Q(state, action)                   │
│      → Apply repetition penalty if action repeated             │
│                                                                 │
│   Q-values: Query QTable(state_id, action_id)                  │
│   Actions: revision=0, easy=1, mixed=2, advanced=3             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. MAP ACTION TO COURSE DESIGN                                  │
├─────────────────────────────────────────────────────────────────┤
│ Map bandit action → course configuration:                       │
│                                                                 │
│   action="revision":                                            │
│      → difficulty = "beginner"                                  │
│      → topics = basic fundamentals (soft reset)                 │
│      → focus on weak areas basics                               │
│                                                                 │
│   action="easy":                                                │
│      → difficulty = "beginner"                                  │
│      → topics = weak areas (gentle)                             │
│                                                                 │
│   action="mixed":                                               │
│      → difficulty = "intermediate"                              │
│      → topics = blend of weak areas + general progression      │
│                                                                 │
│   action="advanced":                                            │
│      → difficulty = "advanced"                                  │
│      → topics = deep dive on weak areas only                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. GENERATE COURSE SKELETON (Fast)                              │
├─────────────────────────────────────────────────────────────────┤
│ POST /api/course/generate                                       │
│   → LLM generates course outline: title, description, modules  │
│   → Create Course in DB:                                        │
│      - user_id, role, title, level, status="draft"             │
│   → Create Module skeletons (M1-M5):                            │
│      - module.title, module.description, module.quiz (JSON)    │
│      - is_completed=False, is_unlocked=(if first)               │
│      - content=None (lazy-loaded)                               │
│   → Return course_id + module list to frontend                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. LAZY-LOAD MODULE CONTENT (On-Demand)                        │
├─────────────────────────────────────────────────────────────────┤
│ GET /api/course/{course_id}/module/{module_id}/content         │
│   → Check DB: if module.content exists → return cached         │
│   → Else:                                                       │
│      - LLM generates markdown content (300-500 words)           │
│      - Update module.content in DB                              │
│      - Return content                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. USER TAKES QUIZ                                              │
├─────────────────────────────────────────────────────────────────┤
│ POST /api/course/{course_id}/module/{module_id}/quiz           │
│   → Evaluate quiz answers (LLM or rule-based)                   │
│   → Create ModuleAttempt: (user_id, module_id, score)          │
│   → Update module.is_completed if score > threshold            │
│   → Unlock next module if current is completed                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. REWARD & RL UPDATE                                           │
├─────────────────────────────────────────────────────────────────┤
│ After course completion:                                        │
│   → calculate_reward(current_score, prev_score, weak_topics)  │
│   → Weighted: 0.5×improvement + 0.3×weak_progress + 0.2×conf  │
│   → update_action_value(state_id, action_id, reward):         │
│      - new_q = (old_q × visit_count + reward) / (count + 1)   │
│      - Store in QTable                                          │
│   → Update UserState for next session                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## KEY FILE DEPENDENCIES

```
Interview Flow:
  main.py → models.py (Interview, InterviewAttempt)
         → core/chains/base_chain.py (InterviewQuestionChain)
         → core/llm/llm_service.py (Groq API)
         → core/prompts/prompt_manager.py (INTERVIEW prompt)
         → services/rag/ (optional context retrieval)
         → speech/transcription.py (Whisper ASR + metrics)

Course Flow:
  main.py → models.py (Course, Module, ModuleAttempt)
         → services/rl/rl_service.py (Bandit action selection)
         → utils/rl_helpers.py (State discretization)
         → core/llm/llm_service.py (Content generation)
         → core/prompts/prompt_manager.py (COURSE prompt)
```

---

## QUICK REFERENCE: WHERE TO FIND THINGS

| What | Where |
|------|-------|
| Interview question generation | [core/chains/base_chain.py](core/chains/base_chain.py) (InterviewQuestionChain) |
| Course generation | [main.py](main.py) (/api/course/generate) + [services/rl/rl_service.py](services/rl/rl_service.py) |
| Resume processing | [main.py](main.py) (/api/upload_resume) + [user_skill_profile.py](user_skill_profile.py) |
| RAG/Vector search | [services/rag/retriever.py](services/rag/retriever.py) |
| LLM integration | [core/llm/llm_service.py](core/llm/llm_service.py) |
| Speech-to-text | [speech/transcription.py](speech/transcription.py) |
| RL bandit logic | [services/rl/rl_service.py](services/rl/rl_service.py) |
| LLM caching | [core/llm/llm_service.py](core/llm/llm_service.py) (LLMCache class) |
| Evaluation metrics | [speech/transcription.py](speech/transcription.py) (analyze_speech_delivery) |
| State discretization | [utils/rl_helpers.py](utils/rl_helpers.py) |
| Prompt templates | [core/prompts/prompt_manager.py](core/prompts/prompt_manager.py) |
| Database schema | [models.py](models.py) |
| Configuration | [config/settings.py](config/settings.py) |

---

## 8. COMPLETE DEPENDENCY MAPS

### **INTERVIEW GENERATION WORKFLOW**

```
FRONTEND
  ↓
  ↓ templates/interview.html
  ↓   (JavaScript captures audio, sends Q&A pairs)
  ↓
  ├─→ 1. Get Next Question
  │
  ├──→ API ENDPOINT
  │    POST /api/interview/next_question
  │
  ├──→ BACKEND FUNCTION (main.py, line 1258)
  │    api_interview_next_question(request, db)
  │      ├─ Extract: user_id, role, level, history
  │      ├─ Build context from:
  │      │  ├─ resume_store[username] (resume text)
  │      │  ├─ interview_sessions[username] (session state)
  │      │  └─ history (previous questions used)
  │      └─ Invoke question_chain.invoke(llm_payload)
  │
  ├──→ LLM CHAIN (core/chains/base_chain.py)
  │    InterviewQuestionChain.invoke()
  │      ├─ Input Validation: Ensure all fields present
  │      ├─ Optional RAG: retriever.retrieve_context()
  │      ├─ Prompt Formatting:
  │      │  └─ prompt_manager.get_prompt("interview", **payload)
  │      │     Returns: System prompt + role/level/resume context
  │      ├─ LLM Call:
  │      │  └─ llm_service.invoke(prompt, json_mode=True)
  │      │     ├─ Check cache (MD5 hash of prompt)
  │      │     ├─ Call Groq API (meta-llama/llama-4-scout-17b)
  │      │     ├─ Parse response JSON
  │      │     └─ Cache result (TTL 3600s)
  │      └─ Return: ChainResult with question JSON
  │
  ├──→ DATABASE ACCESS
  │    No direct DB access (memory-based)
  │      └─ interview_sessions[username] (in-memory dict)
  │         └─ Store question in session["questions"]
  │
  ├──→ AI/RAG COMPONENTS
  │    Optional:
  │      ├─ services/rag/retriever.py:
  │      │  retrieve_context(query, k=5)
  │      │    └─ Search FAISS index for relevant docs
  │      │
  │      └─ services/rag/vector_store.py:
  │         search(query, k)
  │           ├─ Embed query with Sentence-Transformers
  │           ├─ Search FAISS index (L2 distance)
  │           └─ Return top-k documents
  │
  └─→ RESPONSE: {"question": "...", "category": "..."}

───────────────────────────────────────────────────────────

  ├─→ 2. Transcribe Audio
  │
  ├──→ API ENDPOINT
  │    POST /api/transcribe
  │
  ├──→ BACKEND FUNCTION (main.py)
  │    api_transcribe(request)
  │      ├─ Read audio file from request
  │      └─ Call transcribe_audio(audio_file)
  │
  ├──→ SPEECH MODULE (speech/transcription.py)
  │    transcribe_audio(audio_file)
  │      ├─ Load with scipy/librosa
  │      ├─ Pass to OpenAI Whisper model
  │      ├─ Get text output
  │      └─ Call normalize_transcript(text)
  │         └─ Fix STT errors (teh→the, recieve→receive)
  │
  ├──→ DATABASE ACCESS
  │    No DB access (return text to frontend)
  │
  ├──→ AI/RAG COMPONENTS
  │    speech/transcription.py:
  │      └─ Whisper ASR (OpenAI model)
  │
  └─→ RESPONSE: {"transcript": "..."}

───────────────────────────────────────────────────────────

  ├─→ 3. Evaluate All Answers
  │
  ├──→ API ENDPOINT
  │    POST /api/interview/evaluate
  │
  ├──→ BACKEND FUNCTION (main.py, line 1372)
  │    api_interview_evaluate(request, db)
  │      ├─ Extract: role, level, questions_answers
  │      ├─ Call: analyze_speech_delivery() for each answer
  │      ├─ Call: evaluate_content(role, level, questions_answers)
  │      ├─ Aggregate 8 features:
  │      │  1. Candidate profile
  │      │  2. Speech metrics (WPM, fillers, clarity)
  │      │  3. Content scores (LLM)
  │      │  4. Confidence score
  │      │  5. Performance summary
  │      │  6. Overall score & verdict
  │      │  7. Detailed per-answer breakdown
  │      │  8. RL-based course recommendation
  │      └─ Build report JSON
  │
  ├──→ LLM CHAINS (core/chains/base_chain.py)
  │    1. EvaluationChain.invoke() [per Q-A pair]
  │       └─ Score with 5 dimensions:
  │          ├─ Relevance score
  │          ├─ Explanation depth
  │          ├─ STAR method score
  │          ├─ Structured thinking
  │          └─ Problem solving
  │
  │    2. SummaryChain.invoke() [overall performance]
  │       └─ Generate performance summary
  │
  ├──→ DATABASE ACCESS (main.py)
  │    ├─ db.query(Interview).filter() - Query existing interviews
  │    ├─ db.query(UserState).filter() - Get user RL state
  │    ├─ db.add(Interview(...)) - Store new interview
  │    ├─ db.add(UserState(...)) - Update RL state
  │    └─ db.commit() - Persist changes
  │
  │    Models used (models.py):
  │    ├─ Interview: (user_id, role, date, score, report_json)
  │    ├─ UserState: (user_id, state_id, avg_score, session_count)
  │    └─ QTable: (state_id, action_id, q_value, visit_count)
  │
  ├──→ AI/RAG COMPONENTS
  │    1. services/rl/rl_service.py:
  │       ContextualBandit:
  │         ├─ select_action(state_id, user_state, ...)
  │         │  ├─ Cold-start: return "revision" if session_count < 2
  │         │  ├─ ε-greedy: epsilon = 1/(1+session_count)
  │         │  └─ Fetch Q-values from QTable
  │         └─ update_action_value(state_id, action, reward)
  │            └─ Running average: q_new = (q_old×count + reward)/(count+1)
  │
  │    2. utils/rl_helpers.py:
  │       ├─ get_state_id(score, weak_count)
  │       │  └─ Discretize: "low-2", "medium-3", "high-0"
  │       └─ calculate_reward(...)
  │          └─ Weighted: 0.5×improvement + 0.3×weak_progress + 0.2×confidence
  │
  │    3. speech/transcription.py:
  │       ├─ analyze_speech_delivery(answer, duration)
  │       │  └─ Calculate WPM, filler words, clarity score
  │       └─ compute_confidence_score(speech_analyses)
  │
  │    4. create_course_internal() [Triggered by RL]
  │       ├─ Use bandit action to set course difficulty & topics
  │       ├─ LLM generates course outline
  │       └─ Create Course + Module skeletons in DB
  │
  └─→ RESPONSE: 
      {
        "overall_score": 75,
        "verdict": "...",
        "new_course_id": 123,
        "recommended_action": "mixed",
        "rl_metrics": {...}
      }
```

---

### **COURSE GENERATION WORKFLOW**

```
TRIGGER: Interview Evaluation Complete
  ↓
  ↓ api_interview_evaluate() finishes
  ↓
  ├─→ 1. RL Bandit Action Selection
  │
  ├──→ FUNCTION CHAIN
  │    Triggered in: main.py, api_interview_evaluate()
  │      ├─ Extract: overall_score, weak_topics from report
  │      ├─ Call: get_state_id(score, weak_count)
  │      │  Location: utils/rl_helpers.py, line ~20
  │      │  Discretizes: score < 50 → "low", 50-75 → "medium", >75 → "high"
  │      │  Returns: state_id = f"{score_level}-{weak_count}"
  │      ├─ Fetch UserState from DB (models.py: UserState)
  │      ├─ Create: ContextualBandit(db=db, action_space="course")
  │      │  Location: services/rl/rl_service.py, line ~100
  │      └─ Call: select_action(state_id, user_state, ...)
  │         ├─ Cold-start: if session_count < 2 → return "revision"
  │         ├─ ε-greedy: epsilon = 1 / (1 + session_count)
  │         ├─ Query QTable(state_id, action_id) for Q-values
  │         ├─ Explore: random action if rand() < epsilon
  │         ├─ Exploit: argmax Q(state, action) else
  │         └─ Returns: action = "revision"|"easy"|"mixed"|"advanced"
  │
  ├──→ DATABASE ACCESS
  │    db.query(UserState).filter(UserState.user_id == user.id).first()
  │      └─ Models accessed (models.py):
  │         ├─ UserState: (user_id, state_id, avg_score, session_count)
  │         └─ QTable: (state_id, action_id, q_value, visit_count)
  │
  ├──→ LOGGING
  │    utils/bandit_logger.py:
  │      └─ log_bandit_state(...), log_reward_calculation(...), etc.
  │
  └─→ RESULT: action_id in {0: "revision", 1: "easy", 2: "mixed", 3: "advanced"}

───────────────────────────────────────────────────────────

  ├─→ 2. Map Action to Course Topics & Difficulty
  │
  ├──→ FUNCTION (main.py, line 362)
  │    _compose_course_topics(action, weak_topics, ...)
  │      ├─ IF action == "revision":
  │      │  ├─ difficulty = "beginner"
  │      │  └─ topics = basic fundamentals + weak areas
  │      ├─ IF action == "easy":
  │      │  ├─ difficulty = "beginner"
  │      │  └─ topics = weak areas (gentle approach)
  │      ├─ IF action == "mixed":
  │      │  ├─ difficulty = "intermediate"
  │      │  └─ topics = blend weak + general progression
  │      └─ IF action == "advanced":
  │         ├─ difficulty = "advanced"
  │         └─ topics = deep dive weak areas only
  │
  ├──→ NORMALIZE TOPICS (main.py, line 2232)
  │    _normalize_course_topics(role, topics, difficulty, max_topics=6)
  │      └─ Ensure topics are valid, deduplicate, cap at N topics
  │
  └─→ RESULT: 
      {
        "course_topics": ["Design Patterns", "Advanced Python", ...],
        "course_difficulty": "intermediate"
      }

───────────────────────────────────────────────────────────

  ├─→ 3. Generate Course Outline & Skeleton
  │
  ├──→ BACKEND FUNCTION (main.py, line 3055)
  │    create_course_internal(
  │      user, role, level, weak_topics, action, db, 
  │      topics=course_topics, difficulty=course_difficulty
  │    )
  │      ├─ Build outline_prompt_input:
  │      │  ├─ skill: role
  │      │  ├─ level: experience level
  │      │  ├─ bandit_action: selected action
  │      │  ├─ strict_requirements: detailed mapping
  │      │  └─ title_hint: adaptive title suggestion
  │      ├─ Call: prompt_manager.get_prompt("course_outline", **payload)
  │      │  Location: core/prompts/prompt_manager.py
  │      └─ Call: llm_service.invoke(outline_prompt, json_mode=True)
  │         ├─ Location: core/llm/llm_service.py
  │         ├─ Check cache (MD5 hash)
  │         ├─ Call Groq API (meta-llama/llama-4-scout-17b)
  │         ├─ Parse JSON response
  │         ├─ Cache result
  │         └─ Returns: outline_json with course title, description, modules
  │
  ├──→ DATABASE ACCESS (models.py)
  │    ├─ Create Course object:
  │    │  ├─ user_id, role, title, description
  │    │  ├─ level: difficulty (beginner/intermediate/advanced)
  │    │  └─ status: "generated"
  │    │  └─ db.add(course) + db.flush()
  │    │
  │    ├─ Create Module skeletons (M1-M5):
  │    │  ├─ course_id: foreign key to Course
  │    │  ├─ title, description, topics list
  │    │  ├─ quiz: JSON structure (not filled yet)
  │    │  ├─ content: None (lazy-loaded)
  │    │  ├─ is_unlocked: True if first module, False else
  │    │  ├─ is_completed: False
  │    │  └─ db.add(module) for each
  │    │
  │    └─ db.commit() - Persist all
  │
  ├──→ AI/RAG COMPONENTS
  │    1. core/llm/llm_service.py:
  │       LLMService.invoke()
  │         ├─ MD5 hash-based caching
  │         ├─ Groq API call
  │         └─ JSON mode parsing
  │
  │    2. core/prompts/prompt_manager.py:
  │       get_prompt("course_outline", ...)
  │         └─ Returns formatted system + user prompt
  │
  └─→ RESULT: Course created in DB with skeleton modules
      {
        "course_id": 123,
        "modules": [
          {"id": 1, "title": "Module 1", "is_unlocked": true},
          {"id": 2, "title": "Module 2", "is_unlocked": false},
          ...
        ]
      }

───────────────────────────────────────────────────────────

  ├─→ 4. Lazy-Load Module Content (On-Demand)
  │
  ├──→ API ENDPOINT
  │    GET /api/course/{course_id}/module/{module_id}/content
  │
  ├──→ BACKEND FUNCTION
  │    Triggered when user opens module
  │      ├─ Check DB: db.query(Module).filter(Module.id == module_id)
  │      ├─ IF module.content exists:
  │      │  └─ Return cached content immediately
  │      └─ ELSE:
  │         ├─ Build LLM payload (module title, role, difficulty, topics)
  │         ├─ Call: llm_service.invoke(content_prompt, json_mode=False)
  │         ├─ Get markdown content (300-500 words)
  │         ├─ Update: db.query(Module).update({Module.content: content})
  │         └─ db.commit()
  │
  ├──→ DATABASE ACCESS (models.py: Module)
  │    ├─ Query: db.query(Module).filter(Module.id == module_id).first()
  │    ├─ Update: db.query(Module).update({Module.content: generated_content})
  │    └─ db.commit()
  │
  ├──→ AI/RAG COMPONENTS
  │    core/llm/llm_service.py:
  │      └─ LLMService.invoke(content_prompt)
  │         ├─ Check cache
  │         ├─ Groq API call
  │         └─ Return markdown string
  │
  └─→ RESPONSE: {"content": "# Module Title\n\n...markdown..."}

───────────────────────────────────────────────────────────

  ├─→ 5. User Takes Quiz & RL Reward Update
  │
  ├──→ API ENDPOINT
  │    POST /api/course/{course_id}/module/{module_id}/quiz
  │
  ├──→ BACKEND FUNCTION
  │    Triggered when user submits quiz answers
  │      ├─ Evaluate answers (LLM or rule-based)
  │      ├─ Calculate quiz_score
  │      ├─ Create: ModuleAttempt(user_id, module_id, score, answers)
  │      ├─ db.add(module_attempt) + db.commit()
  │      ├─ IF quiz_score > threshold:
  │      │  ├─ Update: Module.is_completed = True
  │      │  └─ Unlock next module: next_module.is_unlocked = True
  │      └─ Return: {"score": quiz_score, "passed": true/false}
  │
  ├──→ DATABASE ACCESS (models.py)
  │    ├─ ModuleAttempt: (user_id, module_id, score, answers)
  │    ├─ Module: UPDATE is_completed, is_unlocked
  │    └─ db.commit()
  │
  └─→ RESULT: Quiz evaluated, progression tracked

───────────────────────────────────────────────────────────

  ├─→ 6. Calculate RL Reward & Update Q-Table
  │
  ├──→ FUNCTION (main.py, inside api_interview_evaluate)
  │    calculate_reward(current_score, prev_score, weak_topics, ...)
  │      Location: utils/rl_helpers.py
  │      Calculation:
  │        ├─ score_improvement = current - previous
  │        ├─ weak_topic_progress = (prev_count - curr_count) / 3.0
  │        ├─ confidence_improvement = (curr_conf - prev_conf) / 100.0
  │        └─ reward = 0.5×score_imp + 0.3×weak_prog + 0.2×conf_imp
  │           └─ Returns: reward in [-1.0, +1.0]
  │
  ├──→ BANDIT Q-VALUE UPDATE (main.py)
  │    course_learner.update_action_value(state_id, action, reward)
  │      Location: services/rl/rl_service.py
  │      Logic:
  │        ├─ Query: QTable(state_id, action_id)
  │        ├─ IF entry exists:
  │        │  ├─ old_q = entry.q_value
  │        │  ├─ visit_count = entry.visit_count
  │        │  └─ new_q = (old_q × visit_count + reward) / (visit_count + 1)
  │        └─ ELSE:
  │           └─ Create new entry with q_value = reward, visit_count = 1
  │        └─ db.update(QTable).set({q_value: new_q, visit_count: count+1})
  │
  ├──→ DATABASE ACCESS (models.py: QTable)
  │    ├─ Query: db.query(QTable).filter(
  │    │           QTable.state_id == state_id,
  │    │           QTable.action_id == action_id
  │    │         )
  │    ├─ Update: db.update(QTable).set({q_value: new_q, visit_count: count+1})
  │    └─ db.commit()
  │
  ├──→ UPDATE USER STATE
  │    db.query(UserState).update({
  │      UserState.state_id: new_state_id,
  │      UserState.avg_score: running_avg,
  │      UserState.session_count: session_count + 1
  │    })
  │
  └─→ RESULT: Q-values & user state updated for next session
      QTable now contains improved action values for future decisions
```

---

### **SUMMARY TABLE: Function Call Chain**

| Step | Frontend → Endpoint | Backend Function | AI/LLM Component | DB Access | Output |
|------|---|---|---|---|---|
| **Interview Q Gen** | interview.html → `/api/interview/next_question` | `api_interview_next_question()` | `InterviewQuestionChain.invoke()` → Groq | `interview_sessions[]` (memory) | `{"question": "..."}` |
| **Transcribe** | interview.html → `/api/transcribe` | `api_transcribe()` | Whisper ASR | None | `{"transcript": "..."}` |
| **Evaluate** | report.html → `/api/interview/evaluate` | `api_interview_evaluate()` | `EvaluationChain`, `SummaryChain` | `Interview`, `UserState`, `QTable` | Report JSON (8 features) |
| **RL Action** | [Internal trigger] | `select_action()` | `ContextualBandit` | `QTable` | Action: "revision"\|"easy"\|"mixed"\|"advanced" |
| **Gen Course** | [Internal trigger] | `create_course_internal()` | `LLMService` (outline) | `Course`, `Module` | `Course` + skeleton `Modules` |
| **Module Content** | course.html → `/api/course/{id}/module/{mid}/content` | [Handler] | `LLMService` (content) | `Module` | Markdown content |
| **Quiz Submit** | module.html → `/api/course/{id}/module/{mid}/quiz` | [Handler] | LLM (eval) or rules | `ModuleAttempt`, `Module` | `{"score": N, "passed": bool}` |
| **RL Reward** | [Internal trigger] | `calculate_reward()`, `update_action_value()` | None | `QTable`, `UserState` | Updated Q-values |
