import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import requests
from bs4 import BeautifulSoup
import json
import time
from tqdm import tqdm
from clean import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_OUTPUT  = PROJECT_ROOT / "data" / "docs.json"

URLS = [
    "https://huggingface.co/docs/transformers/index/",
    "https://huggingface.co/docs/transformers/installation/",
    "https://huggingface.co/docs/transformers/quicktour/",
    "https://huggingface.co/docs/transformers/weightconverter",
    "https://huggingface.co/docs/transformers/models",
    "https://huggingface.co/docs/transformers/custom_models",
    "https://huggingface.co/docs/transformers/monkey_patching",
    "https://huggingface.co/docs/transformers/fusion_mapping",
    "https://huggingface.co/docs/transformers/how_to_hack_models",
    "https://huggingface.co/docs/transformers/model_sharing",
    "https://huggingface.co/docs/transformers/serialization"
]


def normalize_code_block(code: str) -> str:
    lines = code.split("\n")
    stripped = [l.rstrip() for l in lines]

    # Drop leading/trailing blank lines
    while stripped and not stripped[0]:
        stripped.pop(0)
    while stripped and not stripped[-1]:
        stripped.pop()

    if not stripped:
        return ""

    def fix_python_spacing(s: str) -> str:
        import re
        s = re.sub(r'\s*\(\s*', '(', s)
        s = re.sub(r'\s*\)\s*', ')', s)
        s = re.sub(r'\s*\[\s*', '[', s)
        s = re.sub(r'\s*\]\s*', ']', s)
        s = re.sub(r'\s*,\s*', ', ', s)
        s = re.sub(r'\s*:\s*', ': ', s)
        s = re.sub(r'\s*=\s*', '=', s)
        s = re.sub(r'([^=!<>])=([^=])', r'\1 = \2', s)
        s = re.sub(r'==', ' == ', s)
        s = re.sub(r'!=', ' != ', s)
        s = re.sub(r'  +', ' ', s)
        return s.strip()

    if len(stripped) <= 4 and all(len(l.split()) <= 4 for l in stripped if l):
        joined = " ".join(l for l in stripped if l)
        return joined

    fixed = []
    for line in stripped:
        fixed.append(fix_python_spacing(line))
    return "\n".join(fixed)


def extract_main_text(soup):
    main = soup.find("main")
    if not main:
        return ""

    for tag in main.find_all(["nav", "footer", "aside", "script", "style"]):
        tag.decompose()

    for tag in main.find_all(True):
        if tag.get_text(" ", strip=True).startswith("and get access to the augmented"):
            tag.decompose()
            break

    texts = []

    for tag in main.find_all(["h1", "h2", "h3", "p", "li", "pre"]):

        if tag.name == "pre":
            code_tag = tag.find("code")
            raw = (code_tag or tag).get_text("\n")  

            lang = ""
            cls = (code_tag or tag).get("class", [])
            for c in cls:
                if c.startswith("language-"):
                    lang = c.replace("language-", "")
                    break

            normalized = normalize_code_block(raw)
            if not normalized or len(normalized.strip()) < 3:
                continue

            texts.append(f"\n```{lang}\n{normalized}\n```\n")
            continue

        content = tag.get_text(" ", strip=True)

        if len(content) < 30:
            continue

        if tag.name in ["h1", "h2", "h3"]:
            texts.append(f"\n{content}\n")
        elif tag.name == "li":
            texts.append(f"- {content}")
        else:
            texts.append(content)

    return "\n\n".join(texts)


def process_url(url, retries=3):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-doc-fetcher/1.0)"}

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()

            print(f"[SUCCESS] Fetched: {url} (Attempt {attempt+1})\n")

            soup = BeautifulSoup(response.text, "html.parser")

            text = extract_main_text(soup)
            cleaned = clean_text(text)

            title = soup.title.string if soup.title else "No Title"

            return {
                "text": cleaned,
                "metadata": {
                    "source": "huggingface",
                    "url": url,
                    "title": title
                }
            }

        except requests.RequestException as e:
            print(f"[Attempt {attempt+1}/{retries}] Failed: {url}")
            print(f"Error: {e}")

            if attempt < retries - 1:
                print("Waiting 5 seconds before retry...\n")
                time.sleep(5)
            else:
                print(f"[FAILED] All retries failed for: {url}\n")
                return None


def main():
    docs = []

    for i, url in enumerate(tqdm(URLS, desc="Processing URLs")):
        doc = process_url(url)

        if doc and doc["text"].strip():   
            docs.append(doc)

        if i < len(URLS) - 1:
            time.sleep(0.75)

    DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCS_OUTPUT, "w") as f:
        json.dump(docs, f, indent=2)

    print(f"Saved {len(docs)} docs to {DOCS_OUTPUT}")


if __name__ == "__main__":
    main()