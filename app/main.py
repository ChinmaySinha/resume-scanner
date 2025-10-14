# app/main.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.parsers import extract_text_from_bytes, parse_basic_fields
from app.embeddings import embed_text, embed_list, get_embedding_model
from app.skills import SKILLS
from app.scoring import score_with_llm, score_with_cosine
from app.config import settings
import numpy as np

st.set_page_config(page_title="Single Resume Screener (modular)", layout="wide")
st.title("Single-Resume Screener — Modular (LLM + Cosine fallback)")

with st.expander("Instructions"):
    st.write("""
    Upload one resume (.txt or .pdf), paste a job description, and click Process.
    The app first tries to get a score and justification from an LLM (Flan-T5-small).
    If LLM is unavailable or fails, it falls back to cosine-similarity on embeddings.
    """)

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload resume (txt or pdf)", type=["txt", "pdf"])
    job_title = st.text_input("Job title", value="")
    job_desc = st.text_area("Job description", height=200)
    process = st.button("Process / Score")

with col2:
    parsed_box = st.empty()
    score_box = st.empty()
    skills_box = st.empty()
    justification_box = st.empty()
    raw_llm_box = st.empty()

_ = get_embedding_model()

if process:
    if not uploaded:
        st.error("Please upload a resume file.")
    elif not job_desc.strip():
        st.error("Please paste a job description.")
    else:
        data = uploaded.read()
        text = extract_text_from_bytes(data, uploaded.name)
        if not text or not text.strip():
            st.error("Could not extract text from the file. Try a plain .txt resume.")
        else:
            parsed = parse_basic_fields(text)
            parsed_box.subheader("Parsed basic fields")
            parsed_box.json(parsed)

            model = get_embedding_model()
            skill_embs = embed_list(SKILLS)
            resume_vec = embed_text(text)
            sims = (skill_embs @ resume_vec)
            idx = np.argsort(-sims)[:settings.TOP_K_SKILLS]
            top_skills = [{"skill": SKILLS[i], "score": float(sims[i])} for i in idx]
            skills_box.subheader("Top matched skills (from taxonomy)")
            skills_box.table(top_skills)

            with st.spinner("Calling LLM for scoring (can be slow on CPU)..."):
                llm_res = score_with_llm(job_title, job_desc, text)
            if llm_res.get("ok"):
                score_box.subheader(f"Match score (LLM): {llm_res['score']} / 10")
                justification_box.subheader("Justification (LLM)")
                justification_box.write(llm_res.get("justification", ""))
                raw_llm_box.subheader("Raw LLM output (debug)")
                raw_llm_box.code(llm_res.get("raw", "")[:2000])
            else:
                fallback = score_with_cosine(text, job_desc)
                score_box.subheader(f"Match score (cosine fallback): {fallback['score']} / 10")
                justification_box.subheader("Justification (fallback)")
                justification_box.write(f"Cosine similarity: {fallback['similarity']:.4f}. LLM error: {llm_res.get('error')}")
                if llm_res.get("raw"):
                    raw_llm_box.subheader("Raw LLM output (for debugging)")
                    raw_llm_box.code(llm_res.get("raw", "")[:2000])
