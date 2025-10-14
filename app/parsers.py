# app/parsers.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from io import BytesIO
from typing import Dict, Any, List
from pdfminer.high_level import extract_text as extract_pdf_text
import spacy
from spacy.matcher import PhraseMatcher
from transformers import pipeline
from app.config import settings
from . import skills as skills_module

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}")

_nlp = None
_token_classifier = None

def get_spacy():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(settings.SPACY_MODEL)
    return _nlp

def get_token_classifier():
    global _token_classifier
    if _token_classifier is None:
        try:
            _token_classifier = pipeline("token-classification", model=settings.BERT_NER_MODEL, aggregation_strategy="simple", device=-1)
        except Exception as e:
            print(f"[parsers] Warning: could not load token-classifier '{settings.BERT_NER_MODEL}': {e}")
            _token_classifier = None
    return _token_classifier

def extract_text_from_bytes(data: bytes, filename: str = "") -> str:
    name = (filename or "").lower()
    if name.endswith(".txt") or name.endswith(".md"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return str(data)
    if name.endswith(".pdf"):
        try:
            with BytesIO(data) as f:
                text = extract_pdf_text(f)
                return text or ""
        except Exception:
            return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_basic_fields(text: str) -> Dict[str, Any]:
    nlp = get_spacy()
    snippet = text[:15000]
    doc = nlp(snippet)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ in ("ORG",)]
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    return {
        "names": list(dict.fromkeys(names)),
        "orgs": list(dict.fromkeys(orgs)),
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None
    }

# alias expected by main.py
parse_basic_fields = extract_basic_fields

def extract_entities_with_bert(text: str, max_chunk: int = 2000) -> List[Dict[str, Any]]:
    classifier = get_token_classifier()
    if classifier is None:
        return []
    items = []
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i + max_chunk]
        try:
            out = classifier(chunk)
        except Exception as e:
            print(f"[parsers] token-classifier chunk error: {e}")
            out = []
        for ent in out:
            label = ent.get("entity_group") or ent.get("entity") or ent.get("label") or None
            text_ent = ent.get("word") or ent.get("word") or None
            if label and text_ent:
                items.append({
                    "label": label,
                    "text": text_ent,
                    "score": float(ent.get("score", 1.0)),
                    "start": ent.get("start"),
                    "end": ent.get("end")
                })
    seen = set()
    uniq = []
    for e in items:
        key = (e["label"], e["text"].strip().lower())
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq

def extract_skills_from_taxonomy(text: str, taxonomy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    nlp = get_spacy()
    matcher = PhraseMatcher(nlp.vocab)

    # Create patterns for PhraseMatcher
    patterns = {}
    for canonical, variations in taxonomy.items():
        patterns[canonical] = [nlp(variation) for variation in variations]

    for canonical, pattern_list in patterns.items():
        matcher.add(canonical, None, *pattern_list)

    doc = nlp(text)
    matches = matcher(doc)

    found = []
    seen_canonical_skills = set()

    for match_id, start, end in matches:
        canonical_skill = nlp.vocab.strings[match_id]
        if canonical_skill not in seen_canonical_skills:
            found.append({
                "skill": canonical_skill,
                "match": doc[start:end].text,
                "confidence": 1.0
            })
            seen_canonical_skills.add(canonical_skill)

    return found

def extract_sections(text: str) -> Dict[str, str]:
    lower = text.lower()
    sections = {}
    for key, length in (("experience", 4000), ("education", 2000), ("projects", 2000)):
        if key in lower:
            idx = lower.find(key)
            sections[key] = text[idx: idx + length]
    return sections

def parse_resume_text_from_bytes(data: bytes, filename: str = "") -> Dict[str, Any]:
    raw = extract_text_from_bytes(data, filename)
    raw_trim = raw[:20000] if raw else ""
    basic = extract_basic_fields(raw)
    sections = extract_sections(raw)
    bert_entities = extract_entities_with_bert(raw)
    taxonomy_skills = extract_skills_from_taxonomy(raw, skills_module.SKILLS)
    skills = taxonomy_skills
    return {
        "raw": raw_trim,
        "basic": basic,
        "sections": sections,
        "bert_entities": bert_entities,
        "skills": skills
    }
