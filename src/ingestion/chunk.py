import re
import json
from tqdm import tqdm


CHUNK_SIZE = 300   
OVERLAP = 50    
MIN_CHUNK_WORDS = 30 

def split_by_headings(text: str) -> list[str]:

    heading_re = re.compile(r'(?:^|\n)(?=\n?[A-Z][^\n]{2,80}\n)')

    candidate_positions = [m.start() for m in heading_re.finditer(text)]

    safe_positions = [0]
    in_fence = False
    fence_re = re.compile(r'```')
    fence_positions = [m.start() for m in fence_re.finditer(text)]
    fence_idx = 0

    for pos in candidate_positions:
        if pos == 0:
            continue
        while fence_idx < len(fence_positions) and fence_positions[fence_idx] < pos:
            in_fence = not in_fence
            fence_idx += 1
        if not in_fence:
            safe_positions.append(pos)

    safe_positions.append(len(text))
    sections = []
    for i in range(len(safe_positions) - 1):
        chunk = text[safe_positions[i]:safe_positions[i + 1]].strip()
        if chunk:
            sections.append(chunk)

    return sections if sections else [text.strip()]

def split_section_into_parts(section: str) -> list[dict]:
   
    parts = []
    segments = re.split(r'(```[\s\S]*?```)', section)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if seg.startswith("```"):
            parts.append({'type': 'code', 'content': seg})
        else:
            sentences = re.split(r'(?<=[.!?])\s+', seg)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    parts.append({'type': 'text', 'content': sent})
    return parts


def chunk_section(section: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    parts = split_section_into_parts(section)

    chunks: list[str] = []
    current_parts: list[dict] = []   
    current_words = 0

    def flush(current_parts: list[dict]) -> list[str]:
        text = "\n\n".join(p['content'] for p in current_parts).strip()
        return text

    def word_count(s: str) -> int:
        return len(s.split())

    for part in parts:
        wc = word_count(part['content'])
        if part['type'] == 'code':
            if current_words + wc > chunk_size and current_parts:
                chunks.append(flush(current_parts))
                current_parts = []
                current_words = 0

            current_parts.append(part)
            current_words += wc

            if current_words >= chunk_size:
                chunks.append(flush(current_parts))
                current_parts = []
                current_words = 0

            continue

        if current_words + wc > chunk_size and current_parts:
            chunks.append(flush(current_parts))

            overlap_parts: list[dict] = []
            overlap_words = 0
            for prev in reversed(current_parts):
                if prev['type'] == 'code':
                    pw = word_count(prev['content'])
                if overlap_words + pw > overlap:
                    break
                overlap_parts.insert(0, prev)
                overlap_words += pw

            current_parts = overlap_parts
            current_words = overlap_words

        current_parts.append(part)
        current_words += wc

    if current_parts:
        chunks.append(flush(current_parts))

    return chunks

def hybrid_chunk(doc: dict) -> list[str]:
    sections = split_by_headings(doc["text"])
    final_chunks: list[str] = []

    for section in sections:
        chunks = chunk_section(section)
        final_chunks.extend(chunks)

    return final_chunks

def chunk_documents(input_path="data/docs.json", output_path="processed/chunks.json"):
    print("[INFO] Loading documents...\n")

    with open(input_path, "r") as f:
        docs = json.load(f)

    print(f"[INFO] Loaded {len(docs)} documents\n")

    all_chunks = []
    global_id = 0

    for doc in tqdm(docs, desc="Chunking documents"):
        chunks = hybrid_chunk(doc)

        chunks = [c for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS]

        print(f"[DEBUG] {doc['metadata']['title']} → {len(chunks)} chunks")

        for local_id, chunk in enumerate(chunks):
            text = chunk.strip()
            if not text:
                continue

            if text.count("```") % 2 != 0:
                text += "\n```"

            all_chunks.append({
                "text": text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_id": local_id,
                    "global_chunk_id": global_id
                }
            })

            global_id += 1

    wc_list = [len(c["text"].split()) for c in all_chunks]
    print(f"\n[INFO] Total chunks created: {len(all_chunks)}")
    if wc_list:
        print(f"[INFO] Word count — min: {min(wc_list)}, max: {max(wc_list)}, "
              f"mean: {sum(wc_list)//len(wc_list)}, "
              f"median: {sorted(wc_list)[len(wc_list)//2]}")

    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"[SUCCESS] Saved to {output_path}")


if __name__ == "__main__":
    chunk_documents()