# AI-Powered Resume Screener

This project is a sophisticated, AI-powered resume screening application designed to streamline the hiring process. It leverages Natural Language Processing (NLP) and Large Language Models (LLMs) to automatically parse resumes, extract key information, and score candidates against job descriptions.

## Core Features

*   **Resume Parsing:** The application accepts resumes in both `.txt` and `.pdf` formats, automatically extracting the raw text content.
*   **Advanced Skill Extraction:** It uses a powerful skill extraction engine built with spaCy's `PhraseMatcher`. This allows for the identification of a wide range of technical skills, including aliases and variations (e.g., recognizing "k8s" as "Kubernetes"). The skill taxonomy is easily extensible.
*   **AI-Powered Scoring:** The application integrates with Large Language Models (like Google's Gemini or local Flan-T5 models) to provide an intelligent match score (1-10) between a candidate's resume and a given job description. It also provides a brief justification for the score.
*   **Fallback Mechanism:** In case the primary LLM fails, the system gracefully falls back to a cosine similarity score based on text embeddings, ensuring a reliable screening process.
*   **Interactive Frontend:** The user interface is built with Streamlit, providing a simple and intuitive web-based dashboard for uploading resumes, entering job descriptions, and viewing the screening results.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file in the root of the project and add the following variables:
    ```
    # LLM_MODE can be "GEMINI" or "LOCAL"
    LLM_MODE=LOCAL

    # If using GEMINI, provide your API key
    GEMINI_API_KEY=your-gemini-api-key
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app/main.py
    ```

2.  **Open the application in your browser.** The URL will be displayed in your terminal (usually `http://localhost:8501`).

3.  **Upload a resume, enter a job description, and click "Process / Score"** to see the results.
