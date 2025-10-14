# llm.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import json
from app.config import settings
from transformers import pipeline

# Optional: import Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None


def get_local_llm():
    """Load a local model if configured."""
    model_name = settings.LLM_MODEL or "google/flan-t5-small"
    return pipeline("text2text-generation", model=model_name)


def call_local(prompt: str):
    """Generate output using local Flan-T5 model."""
    model = get_local_llm()
    res = model(prompt, max_new_tokens=200)
    return res[0]["generated_text"]


def call_gemini(prompt: str):
    """Generate output using Google Gemini API."""
    if not genai:
        raise ImportError("google-generativeai not installed. Run `pip install google-generativeai`")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def get_score_with_llm(job_title, job_desc, resume_text):
    """Main scoring function."""
    resume_trim = resume_text[:15000]   # 15000 chars max
    job_trim = job_desc[:8000]
    prompt = f"""
    You are an AI resume screener. Compare this resume to the job description.
    Return a JSON object with:
    {{
      "score": <integer 1-10>,
      "justification": "<brief reason>"
    }}

    Job Title: {job_title}
    Job Description: {job_trim}
    Resume: {resume_trim}
    """

    try:
        if settings.LLM_MODE.upper() == "GEMINI":
            try:
                output = call_gemini(prompt)
            except Exception as gem_err:
                print(f"[WARN] Gemini failed: {gem_err}. Falling back to local LLM.")
                output = call_local(prompt)
        else:
            output = call_local(prompt)
            


        # Try to extract JSON from model output
        start, end = output.find("{"), output.rfind("}")
        data = json.loads(output[start:end+1]) if start != -1 else {}
        score = data.get("score") or 0
        justification = data.get("justification") or output
        return {"ok": True, "score": int(score), "justification": justification, "raw": output}

    except Exception as e:
        return {"ok": False, "error": str(e), "raw": ""}

    
