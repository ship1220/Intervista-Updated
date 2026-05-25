# user_skill_profile.py
# Pydantic models used for user skill profiling and interview tracking

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

try:
    from speech.transcription import normalize_transcript
except ImportError:
    def normalize_transcript(text: str) -> str:
        return (text or "").strip()


# ============================================================
# BASIC USER INFO
# ============================================================

class BasicUserInfo(BaseModel):

    user_id: str
    name: str
    target_role: str
    experience_level: str
    resume_file_path: str

    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# RESUME STRUCTURE
# ============================================================

class Project(BaseModel):

    title: str
    description: str


class Education(BaseModel):

    degree: str
    university: str
    graduation_year: int


class WorkExperience(BaseModel):

    company: str
    role: str
    duration: str


class ResumeData(BaseModel):

    skills: List[str] = Field(default_factory=list)

    projects: List[Project] = Field(default_factory=list)

    education: Optional[Education] = None

    work_experience: List[WorkExperience] = Field(default_factory=list)


# ============================================================
# SKILL GRAPH NODE
# ============================================================

class SkillNode(BaseModel):

    skill_name: str

    proficiency_level: str = "beginner"

    score: float = Field(0, ge=0, le=100)

    last_updated: datetime = Field(default_factory=datetime.utcnow)

    times_tested: int = 0

    @field_validator("proficiency_level")
    def validate_level(cls, v):

        allowed = {"beginner", "intermediate", "advanced"}

        if v not in allowed:
            raise ValueError("Invalid proficiency level")

        return v


# ============================================================
# TECHNICAL SKILLS
# ============================================================

class TechnicalSkillVector(BaseModel):

    dsa: float = Field(50.0, ge=0, le=100)

    dbms: float = Field(50.0, ge=0, le=100)

    operating_systems: float = Field(50.0, ge=0, le=100)

    computer_networks: float = Field(50.0, ge=0, le=100)

    system_design: float = Field(50.0, ge=0, le=100)


# ============================================================
# INTERVIEW SKILLS
# ============================================================

class InterviewSkillVector(BaseModel):

    relevance: float = Field(50.0, ge=0, le=100)

    explanation_depth: float = Field(50.0, ge=0, le=100)

    structured_thinking: float = Field(50.0, ge=0, le=100)

    problem_solving: float = Field(50.0, ge=0, le=100)

    star_method: float = Field(50.0, ge=0, le=100)


# ============================================================
# COMMUNICATION SKILLS
# ============================================================

class CommunicationSkillVector(BaseModel):

    clarity: float = Field(50.0, ge=0, le=100)

    confidence: float = Field(50.0, ge=0, le=100)

    engagement: float = Field(50.0, ge=0, le=100)

    speaking_pace: float = Field(50.0, ge=0, le=100)

    filler_control: float = Field(50.0, ge=0, le=100)


# ============================================================
# COMPLETE SKILL VECTOR
# ============================================================

class UserSkillVector(BaseModel):

    technical_skills: TechnicalSkillVector = Field(default_factory=TechnicalSkillVector)

    interview_skills: InterviewSkillVector = Field(default_factory=InterviewSkillVector)

    communication_skills: CommunicationSkillVector = Field(default_factory=CommunicationSkillVector)


# ============================================================
# INTERVIEW SCORING
# ============================================================

class ScoreBreakdown(BaseModel):

    correctness: float = Field(ge=0, le=100)

    conceptual_depth: float = Field(ge=0, le=100)

    clarity: float = Field(ge=0, le=100)

    feedback: str

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InterviewRecord(BaseModel):

    question: str

    topic: str

    answer_transcript: str

    evaluation_score: float = Field(ge=0, le=100)

    score_breakdown: ScoreBreakdown


# ============================================================
# COURSE TRACKING
# ============================================================

class CourseProgress(BaseModel):

    course_name: str

    completion_percentage: float = Field(0, ge=0, le=100)

    quizzes_completed: int = 0

    last_accessed: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# MAIN USER SKILL PROFILE
# ============================================================

class UserSkillProfile(BaseModel):

    user_id: str

    basic_info: BasicUserInfo

    resume_data: ResumeData

    technical_skills: TechnicalSkillVector = Field(default_factory=TechnicalSkillVector)

    interview_skills: InterviewSkillVector = Field(default_factory=InterviewSkillVector)

    communication_skills: CommunicationSkillVector = Field(default_factory=CommunicationSkillVector)

    overall_score: float = Field(50, ge=0, le=100)

    interview_count: int = 0

    interview_history: List[InterviewRecord] = Field(default_factory=list)

    courses: List[CourseProgress] = Field(default_factory=list)

    weak_topics: List[str] = Field(default_factory=list)

    last_updated: datetime = Field(default_factory=datetime.utcnow)


    # ============================================================
# PROFILE HELPERS (required by main.py)
# ============================================================

def create_user_profile(basic_info: BasicUserInfo, resume_data: ResumeData) -> UserSkillProfile:
    """
    Initialize a new user skill profile.
    """
    return UserSkillProfile(
        user_id=basic_info.user_id,
        basic_info=basic_info,
        resume_data=resume_data,
    )


def detect_weaknesses(profile: UserSkillProfile) -> list[str]:
    """
    Detect weak skills below threshold and update profile.
    """
    weaknesses = []

    if not hasattr(profile, 'technical_skills') or profile.technical_skills is None:
        return weaknesses

    try:
        for name, value in profile.technical_skills.model_dump().items():
            if value < 40:
                weaknesses.append(name)
    except Exception:
        pass

    try:
        for name, value in profile.interview_skills.model_dump().items():
            if value < 40:
                weaknesses.append(name)
    except Exception:
        pass

    try:
        for name, value in profile.communication_skills.model_dump().items():
            if value < 40:
                weaknesses.append(name)
    except Exception:
        pass

    # Update profile weak_topics
    if weaknesses:
        profile.weak_topics = list(set(list(profile.weak_topics) + weaknesses))

    return weaknesses


def recommend_micro_courses(profile: UserSkillProfile) -> list[dict]:
    """
    Generate micro-learning suggestions based on weak topics.
    """
    weak_topics = getattr(profile, 'weak_topics', [])
    if not weak_topics:
        weak_topics = []
    
    return [
        {
            "topic": topic,
            "course": f"Practice and revise {topic}",
        }
        for topic in weak_topics if isinstance(topic, str) and topic.strip()
    ]


_STRUCTURE_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "it", "this", "that", "what", "how", "why", "when", "your",
}


def score_answer_structure(answer: str, question: str) -> dict[str, float]:
    """
    Heuristic dimension scores from transcript text (complements LLM scores).
    STAR, structure, problem-solving, and question relevance.
    """
    text = normalize_transcript(answer)
    low = text.lower()
    empty = not text or low in ("(skipped)", "(no response)", "")

    if empty:
        return {
            "relevance": 0.0,
            "explanation_depth": 0.0,
            "star_method": 0.0,
            "structured_thinking": 0.0,
            "problem_solving": 0.0,
        }

    words = low.split()
    word_count = len(words)

    star_groups = (
        ["situation", "context", "when i", "while working", "at my company", "project"],
        ["task", "responsibility", "goal", "needed to", "challenge was"],
        ["i built", "i implemented", "i led", "i designed", "my approach", "we used", "i created"],
        ["result", "outcome", "impact", "improved", "reduced", "increased", "saved", "achieved"],
    )
    star_hits = sum(1 for group in star_groups if any(p in low for p in group))
    star_method = min(100.0, 15.0 * star_hits + min(25.0, word_count / 4))

    structure_markers = [
        "first", "second", "then", "next", "finally", "because", "therefore",
        "step", "approach", "framework", "for example", "in summary",
    ]
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    structure_hits = sum(1 for m in structure_markers if m in low)
    structured_thinking = min(
        100.0, 25.0 + 8.0 * structure_hits + min(25.0, len(sentences) * 4)
    )

    problem_markers = [
        "solve", "solution", "debug", "fix", "optimize", "trade-off", "tradeoff",
        "complexity", "edge case", "alternative", "bottleneck", "root cause",
        "algorithm", "design", "architecture",
    ]
    problem_hits = sum(1 for m in problem_markers if m in low)
    problem_solving = min(100.0, 30.0 + 10.0 * problem_hits + min(20.0, word_count / 5))

    q_tokens = {
        t for t in re.findall(r"[a-z0-9]+", (question or "").lower())
        if len(t) > 2 and t not in _STRUCTURE_STOP
    }
    a_tokens = {t for t in re.findall(r"[a-z0-9]+", low) if len(t) > 2}
    overlap = (len(q_tokens & a_tokens) / len(q_tokens)) if q_tokens else 0.5
    relevance = min(100.0, 35.0 + 55.0 * overlap)

    explanation_depth = min(100.0, 20.0 + word_count * 0.45 + 5.0 * len(sentences))

    return {
        "relevance": round(relevance, 1),
        "explanation_depth": round(explanation_depth, 1),
        "star_method": round(star_method, 1),
        "structured_thinking": round(structured_thinking, 1),
        "problem_solving": round(problem_solving, 1),
    }


def blend_dimension_score(
    llm_value,
    heuristic_value: float,
    heuristic_weight: float = 0.4,
) -> float:
    """Merge LLM and heuristic dimension scores."""
    try:
        llm = float(llm_value)
    except (TypeError, ValueError):
        return round(float(heuristic_value), 1)
    h = float(heuristic_value)
    blended = (1.0 - heuristic_weight) * llm + heuristic_weight * h
    return round(max(0.0, min(100.0, blended)), 1)


def aggregate_dimension_scores(per_answer_dims: list[dict]) -> dict[str, float]:
    """Average per-answer dimension scores across an interview."""
    keys = (
        "relevance",
        "explanation_depth",
        "star_method",
        "structured_thinking",
        "problem_solving",
    )
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(d[key]) for d in per_answer_dims if key in d and d[key] is not None]
        if vals:
            out[key] = round(sum(vals) / len(vals), 1)
    return out


def _skill_tier(value: float) -> str:
    if value >= 70:
        return "strong"
    if value >= 55:
        return "developing"
    return "weak"


def _focus_action_for(skill_key: str, score: float) -> str:
    actions = {
        "star_method": "Practice STAR: Situation → Task → Action → Result on behavioral questions.",
        "structured_thinking": "Outline answers: context, approach, steps, outcome before speaking.",
        "problem_solving": "Explain trade-offs, alternatives, and why you chose your approach.",
        "relevance": "Anchor each answer to the question keywords in the first sentence.",
        "explanation_depth": "Add concrete examples, metrics, and technical detail.",
        "clarity": "Use shorter sentences and pause between key points.",
        "confidence": "Reduce filler words; prepare 2–3 bullet points before answering.",
        "engagement": "Vary vocabulary and show enthusiasm for the topic.",
        "speaking_pace": "Aim for 120–160 words per minute; avoid rushing.",
        "filler_control": "Replace um/like with brief pauses.",
    }
    return actions.get(
        skill_key,
        f"Targeted practice to raise this skill above 60% (currently {score:.0f}%).",
    )


def build_dashboard_analytics(profile_data: dict, interviews: list) -> dict:
    """
    Distinct dashboard sections:
    - snapshot: high-level KPIs only
    - analysis: full dimension breakdown with tiers
    - focus_areas: weak items + one-line actions only
    """
    interview_count = int(profile_data.get("interview_count") or len(interviews) or 0)
    has_data = interview_count > 0 and bool(interviews)

    if not has_data:
        return {
            "has_data": False,
            "snapshot": [],
            "analysis": [],
            "focus_areas": [],
        }

    inter = profile_data.get("interview_skills") or {}
    comm = profile_data.get("communication_skills") or {}
    overall = float(profile_data.get("overall_score") or 0)

    last_score = None
    if interviews:
        latest = max(interviews, key=lambda iv: iv.date or datetime.min)
        last_score = round(float(latest.score), 1) if latest.score is not None else None

    inter_vals = [float(v) for v in inter.values() if v is not None]
    comm_vals = [float(v) for v in comm.values() if v is not None]
    inter_avg = round(sum(inter_vals) / len(inter_vals), 1) if inter_vals else None
    comm_avg = round(sum(comm_vals) / len(comm_vals), 1) if comm_vals else None

    snapshot = []
    if overall:
        snapshot.append({"label": "Overall readiness", "value": round(overall, 1), "kind": "primary"})
    if last_score is not None:
        snapshot.append({"label": "Latest interview", "value": last_score, "kind": "score"})
    snapshot.append({"label": "Sessions completed", "value": interview_count, "kind": "count"})
    if inter_avg is not None:
        snapshot.append({"label": "Content skills (avg)", "value": inter_avg, "kind": "metric"})
    if comm_avg is not None:
        snapshot.append({"label": "Delivery (avg)", "value": comm_avg, "kind": "metric"})

    dimension_specs = [
        ("interview_skills", "relevance", "Answer relevance"),
        ("interview_skills", "explanation_depth", "Explanation depth"),
        ("interview_skills", "structured_thinking", "Structured thinking"),
        ("interview_skills", "problem_solving", "Problem solving"),
        ("interview_skills", "star_method", "STAR method"),
        ("communication_skills", "clarity", "Clarity"),
        ("communication_skills", "confidence", "Confidence"),
        ("communication_skills", "engagement", "Engagement"),
        ("communication_skills", "speaking_pace", "Speaking pace"),
        ("communication_skills", "filler_control", "Filler control"),
    ]

    analysis = []
    for group, key, label in dimension_specs:
        raw = (profile_data.get(group) or {}).get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        tier = _skill_tier(val)
        analysis.append({
            "key": key,
            "label": label,
            "value": round(val, 1),
            "tier": tier,
            "group": "Interview" if group == "interview_skills" else "Communication",
        })

    analysis.sort(key=lambda x: x["value"])

    focus_areas = []
    for row in analysis:
        if row["tier"] != "weak":
            continue
        focus_areas.append({
            "name": row["label"],
            "score": row["value"],
            "action": _focus_action_for(row["key"], row["value"]),
        })
    focus_areas = focus_areas[:5]

    return {
        "has_data": True,
        "snapshot": snapshot,
        "analysis": analysis,
        "focus_areas": focus_areas,
    }


def update_skill_score(current_score: float, new_score: float) -> float:
    """
    Update skill score using weighted averaging.
    """
    return round((current_score * 0.7) + (new_score * 0.3), 2)


def update_skill_vector(skill_vector: UserSkillVector, updates: dict) -> UserSkillVector:

    for key, value in updates.items():

        if hasattr(skill_vector.technical_skills, key):
            current = getattr(skill_vector.technical_skills, key)
            setattr(
                skill_vector.technical_skills,
                key,
                update_skill_score(current, value)
            )

        elif hasattr(skill_vector.interview_skills, key):
            current = getattr(skill_vector.interview_skills, key)
            setattr(
                skill_vector.interview_skills,
                key,
                update_skill_score(current, value)
            )

        elif hasattr(skill_vector.communication_skills, key):
            current = getattr(skill_vector.communication_skills, key)
            setattr(
                skill_vector.communication_skills,
                key,
                update_skill_score(current, value)
            )

    return skill_vector

def calculate_overall_score(skill_vector: UserSkillVector) -> float:
    """
    Compute overall skill score.
    """

    tech = sum(skill_vector.technical_skills.model_dump().values()) / 5
    interview = sum(skill_vector.interview_skills.model_dump().values()) / 5
    comm = sum(skill_vector.communication_skills.model_dump().values()) / 5

    return round((tech * 0.5) + (interview * 0.3) + (comm * 0.2), 2)


def record_interview_result(profile: UserSkillProfile, interview: InterviewRecord):
    """
    Store interview record and update stats.
    """

    profile.interview_history.append(interview)
    profile.interview_count += 1
    profile.last_updated = datetime.utcnow()

    return profile