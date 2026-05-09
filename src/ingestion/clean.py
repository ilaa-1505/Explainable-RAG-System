import re


def normalize_text(text: str) -> str:
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\.\s+([a-zA-Z0-9])', r'.\1', text)
    text = re.sub(r'(\w)\s*/\s*(\w)', r'\1/\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s*```\s*', '\n```\n', text)
    text = re.sub(r'(?<!\n)(uv pip install)', r'\n\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = normalize_text(text)   
    return text.strip()