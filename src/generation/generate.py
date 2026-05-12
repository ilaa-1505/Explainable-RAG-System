import requests
from src.retrieval.query import retrieve
import os
from dotenv import load_dotenv

load_dotenv()

def build_prompt(query, context):
    return f"""You are a helpful assistant. Answer using ONLY the context below. Do NOT use outside knowledge.

### Context
{context}

### Question
{query}

### Instructions

- Start directly with the answer. No introduction.
- Use only information from the context.
- Do NOT invent or modify commands beyond fixing formatting.

---

### Structure

- If the question is conceptual or descriptive (e.g. "what is X", "explain X") → 
  explain fully using all relevant information from the context. Cover features, 
  use cases, and design principles if present. Do NOT use methods or numbered steps.

- If there is ONE clear workflow → use numbered steps.

- Only treat something as a separate method if the context explicitly presents it as a distinct approach to solving the same task.
- Do NOT create methods from examples, helper classes, or unrelated concepts.
- If the context describes only ONE workflow or approach, DO NOT create multiple methods.
- In that case, use numbered steps instead.

- If there are MULTIPLE valid ways to perform the task:
  → group them into separate sections using:

### Method 1 - Name
### Method 2 - Name

- Each method must be self-contained.
- Do NOT mix commands from different methods.
- Do NOT merge all methods into one numbered list.

---

### Code Rules

- All commands MUST be inside fenced code blocks.
- Use:
  - ```bash for shell commands
  - ```python for Python code

- Commands must be valid and executable.

- Fix formatting issues:
  - Merge broken tokens (". env" → ".env")
  - Fix paths ("source . env /bin/activate" → "source .env/bin/activate")
  - Split combined commands into separate lines

- NEVER:
  - put multiple commands on the same line
  - leave commands outside code blocks
  - use inline code blocks like ```bash command```

---

### Output Requirements

- Output must be clean, valid markdown.
- Code blocks must render correctly.

### Answer
"""

import requests
import os

def generate_answer(prompt, model="llama-3.1-8b-instant"):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "max_tokens": 2048,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
    )
    data = response.json()
    print("GROQ RESPONSE:", data) 

    return response.json()["choices"][0]["message"]["content"]

def build_context(docs, metas, scores):
    context = ""

    for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
        context += f"""
                [Source {i+1} | Score: {round(score, 3)} | {meta.get('url', 'N/A')}]
                Title: {meta.get("title", "N/A")}

                {doc}

                {"-"*50}
                """
    return context




