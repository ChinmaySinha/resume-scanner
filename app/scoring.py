# # app/scoring.py
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import json
# from typing import Optional, Dict, Any
# from app.llm import get_score_with_llm
# from app.embeddings import embed_text
# import numpy as np
# from app.config import settings

# def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
#     if a is None or b is None:
#         return 0.0
#     if a.shape != b.shape:
#         return 0.0
#     denom = (np.linalg.norm(a) * np.linalg.norm(b))
#     if denom == 0:
#         return 0.0
#     return float(np.dot(a, b) / denom)

# def build_prompt(job_title: str, job_desc: str, resume_text: str, max_resume_chars: int = 4000) -> str:
#     resume_trim = (resume_text or "")[:max_resume_chars]
#     prompt = (
#         "You are an expert hiring assistant. Compare the resume below with the job description. "
#         "Return ONLY valid JSON with keys: score (a number between 1 and 10) and justification (a short explanation).\n\n"
#         f"Job Title: {job_title}\n"
#         f"Job Description: {job_desc}\n\n"
#         f"Candidate Resume: {resume_trim}\n\n"
#         "Return output as strict JSON like: {\"score\": 7.5, \"justification\": \"Has Python but lacks AWS experience.\"}"
#     )
#     return prompt

# def parse_llm_json(raw_text: str) -> Optional[Dict[str, Any]]:
#     """
#     Find first {...} JSON in raw_text and parse it. Return dict or None.
#     """
#     try:
#         first = raw_text.find("{")
#         last = raw_text.rfind("}")
#         if first != -1 and last != -1 and last > first:
#             j = raw_text[first:last+1]
#             parsed = json.loads(j)
#             return parsed
#         # fallback try parse entire output
#         parsed = json.loads(raw_text)
#         return parsed
#     except Exception:
#         return None

# def score_with_llm(job_title: str, job_desc: str, resume_text: str) -> Dict[str, Any]:
#     prompt = build_prompt(job_title, job_desc, resume_text)
#     res = call_llm(prompt)
#     if not res.get("success"):
#         return {"ok": False, "error": res.get("error"), "fallback": None}
#     raw = res.get("raw", "")
#     parsed = parse_llm_json(raw)
#     if parsed and "score" in parsed:
#         try:
#             score = float(parsed["score"])
#             if score < 1: score = 1.0
#             if score > 10: score = 10.0
#             justification = str(parsed.get("justification", parsed.get("explanation", "")))
#             return {"ok": True, "score": round(score,2), "justification": justification, "raw": raw, "time": res.get("time")}
#         except Exception:
#             return {"ok": False, "error": "invalid_score_value", "raw": raw}
#     else:
#         return {"ok": False, "error": "json_parse_failed", "raw": raw}

# def score_with_cosine(resume_text: str, job_desc: str) -> Dict[str, Any]:
#     r_vec = embed_text(resume_text)
#     j_vec = embed_text(job_desc)
#     sim = cosine_sim(r_vec, j_vec)
#     sim_clamped = max(0.0, min(1.0, sim))
#     score = round(1 + 9 * sim_clamped, 2)
#     return {"score": score, "similarity": float(sim_clamped)}


# app/scoring.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import Optional, Dict, Any
from app.llm import get_score_with_llm
from app.embeddings import embed_text
import numpy as np
from app.config import settings


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_prompt(job_title: str, job_desc: str, resume_text: str, max_resume_chars: int = 4000) -> str:
    resume_trim = (resume_text or "")[:max_resume_chars]
    prompt = (
        "You are an expert hiring assistant. Compare the resume below with the job description. "
        "Return ONLY valid JSON with keys: score (1–10) and justification (short explanation).\n\n"
        f"Job Title: {job_title}\n"
        f"Job Description: {job_desc}\n\n"
        f"Candidate Resume: {resume_trim}\n\n"
        "Return output as strict JSON like: "
        "{\"score\": 7.5, \"justification\": \"Has Python but lacks AWS experience.\"}"
    )
    return prompt


def parse_llm_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from model output."""
    try:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            parsed = json.loads(raw_text[first:last+1])
            return parsed
        return json.loads(raw_text)
    except Exception:
        return None


def score_with_llm(job_title: str, job_desc: str, resume_text: str) -> Dict[str, Any]:
    """
    Ask Gemini or local Flan model for a resume match score + justification.
    Falls back to local if Gemini fails (handled inside llm.py).
    """
    result = get_score_with_llm(job_title, job_desc, resume_text)

    if not result.get("ok"):
        return {"ok": False, "error": result.get("error"), "raw": result.get("raw")}

    score = float(result.get("score", 0))
    justification = result.get("justification", "")
    raw = result.get("raw", "")

    # Clamp score 1–10
    score = min(max(score, 1.0), 10.0)
    return {
        "ok": True,
        "score": round(score, 2),
        "justification": justification,
        "raw": raw,
    }


def score_with_cosine(resume_text: str, job_desc: str) -> Dict[str, Any]:
    """Simple cosine similarity fallback scoring."""
    r_vec = embed_text(resume_text)
    j_vec = embed_text(job_desc)
    sim = cosine_sim(r_vec, j_vec)
    sim_clamped = max(0.0, min(1.0, sim))
    score = round(1 + 9 * sim_clamped, 2)
    return {"score": score, "similarity": float(sim_clamped)}

