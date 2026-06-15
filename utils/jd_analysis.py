# utils/jd_analysis.py
# Lightweight resume vs job description skill gap analysis for interviews.

import json
import re
from typing import Any, Dict, List, Optional


def _empty_jd_result() -> Dict[str, Any]:
    return {"matched_skills": [], "missing_skills": [], "ats_score": None}


def _normalize_skill_list(items: Any, limit: int = 12) -> List[str]:
    if not isinstance(items, list):
        return []
    normalized = []
    seen = set()
    for item in items:
        skill = str(item).strip()
        if not skill:
            continue
        key = skill.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(skill)
        if len(normalized) >= limit:
            break
    return normalized


def _heuristic_ats_score(resume_text: str, job_description: str) -> int:
    """Approximate ATS overlap when LLM parsing is unavailable."""
    resume_tokens = set(re.findall(r"[a-z][a-z0-9+#.]{1,}", resume_text.lower()))
    jd_tokens = set(re.findall(r"[a-z][a-z0-9+#.]{1,}", job_description.lower()))
    stop = {
        "and", "the", "with", "for", "you", "your", "will", "have", "our",
        "this", "that", "from", "are", "was", "were", "able", "work", "team",
        "role", "job", "experience", "years", "skills", "required", "preferred",
    }
    jd_keywords = {t for t in jd_tokens if len(t) > 2 and t not in stop}
    if not jd_keywords:
        return 50
    overlap = len(jd_keywords & resume_tokens)
    ratio = overlap / max(len(jd_keywords), 1)
    return max(0, min(100, int(round(ratio * 100))))


async def analyze_resume_vs_jd(
    resume_text: str,
    job_description: str,
    *,
    llm_service,
    prompt_manager,
) -> Dict[str, Any]:
    """
    Compare resume against a job description and return skill gap signals.

    If job_description is empty, returns null ATS score and empty skill lists.
    """
    jd = (job_description or "").strip()
    if not jd:
        return _empty_jd_result()

    resume = (resume_text or "").strip()
    prompt = prompt_manager.get_prompt(
        "resume_jd_skill_gap",
        resume_text=resume[:6000] if resume else "No resume text provided.",
        job_description=jd[:8000],
    )

    try:
        raw = await llm_service.invoke(prompt, json_mode=True, use_cache=False)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise ValueError("Invalid JD analysis response")

        matched = _normalize_skill_list(parsed.get("matched_skills", []))
        missing = _normalize_skill_list(parsed.get("missing_skills", []))

        ats_score = parsed.get("ats_score")
        try:
            ats_score = int(round(float(ats_score))) if ats_score is not None else None
        except (TypeError, ValueError):
            ats_score = _heuristic_ats_score(resume, jd)
        if ats_score is not None:
            ats_score = max(0, min(100, ats_score))

        if ats_score is None:
            ats_score = _heuristic_ats_score(resume, jd)

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "ats_score": ats_score,
        }
    except Exception:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "ats_score": _heuristic_ats_score(resume, jd),
        }
