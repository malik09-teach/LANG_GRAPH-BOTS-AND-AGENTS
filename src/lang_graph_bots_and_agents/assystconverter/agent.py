# agent.py
import re
import os 
import requests
import pdfplumber
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY") or ""
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT") or ""
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY") or ""
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or ""


GROQ_AVAILABLE = ChatGroq(model="openai/gpt-oss-120b")


SYSTEM_PROMPT = """You are an expert exam parser. Your job is to extract questions and answers from the given text and format them strictly into ASSIST/Assyst test format.

Rules:
1. Every question must start with '?' on the first line. Do NOT include numbers like "1." in the question text.
2. Every correct answer must start with '+ ' on a new line.
3. Every incorrect answer must start with '- ' on a new line.
4. Remove option letters (A., B., C., D., 1., 2., etc.) and trailing punctuation (semicolons, periods at the end of option lines).
5. Separate consecutive questions with a blank line.
6. If the text has answers marked with color, asterisk (*), bold, or an answer key, mark the correct one with '+'. If uncertain, select the most medically/factually accurate answer as '+'.
7. Output ONLY the formatted questions. Do not add markdown commentary, intro, or outro.
"""

def is_red_char(char):
    col = char.get("non_stroking_color")
    if col == (1.0, 0.0, 0.0) or col == [1.0, 0.0, 0.0]:
        return True
    if isinstance(col, (list, tuple)) and len(col) >= 3:
        if col[0] > 0.7 and col[1] < 0.3 and col[2] < 0.3:
            return True
    return False

def parse_with_smart_color(pdf_file):
    all_lines = []
    with pdfplumber.open(pdf_file) as pdf:
        for p_idx, page in enumerate(pdf.pages):
            words = page.extract_words(extra_attrs=["non_stroking_color"])
            lines, curr_line, curr_top = [], [], None
            for w in words:
                if w["top"] < 40:
                    continue
                if not curr_line:
                    curr_line.append(w)
                    curr_top = w["top"]
                elif abs(w["top"] - curr_top) < 3.5:
                    curr_line.append(w)
                else:
                    lines.append(curr_line)
                    curr_line = [w]
                    curr_top = w["top"]
            if curr_line:
                lines.append(curr_line)

            for l in lines:
                text = " ".join([w["text"] for w in l])
                has_red = any(is_red_char(w) for w in l)
                all_lines.append({"text": text, "has_red": has_red, "page": p_idx + 1})

    opt_pattern = re.compile(r"^\s*([A-Za-z]\.|\b[A-Za-z]\)|\b\d+[\.\)]|#&?)\s*(.*)", re.IGNORECASE)
    output_blocks = []
    current_q_text, current_options, curr_opt, in_options = [], [], None, False

    def flush_question():
        if not current_q_text or not current_options:
            return
        q_head = "?" + " ".join(current_q_text).strip()
        opts = []
        for o in current_options:
            prefix = "+" if o["is_correct"] else "-"
            cleaned_text = re.sub(r"[;\.]+$", "", o["text"]).strip()
            opts.append(f"{prefix} {cleaned_text}")
        output_blocks.append(q_head + "\n" + "\n".join(opts))

    for line in all_lines:
        t = line["text"].strip()
        if not t:
            continue

        m_q = re.match(r"^(@|\d+[\.\)])\s*(.*)", t)
        if m_q and (len(current_options) >= 2 or not current_q_text):
            flush_question()
            current_q_text = [m_q.group(2).strip()]
            current_options, curr_opt, in_options = [], None, False
            continue

        m_opt = opt_pattern.match(t)
        if m_opt:
            in_options = True
            if curr_opt:
                current_options.append(curr_opt)
            rest = re.sub(r"^(\$[A-Za-z][\.\)]\s*|[A-Za-z][\.\)]\s*)+", "", m_opt.group(2).strip()).strip()
            curr_opt = {"text": rest, "is_correct": line["has_red"]}
        else:
            if not in_options:
                current_q_text.append(t)
            else:
                if curr_opt:
                    curr_opt["text"] += " " + t
                    if line["has_red"]:
                        curr_opt["is_correct"] = True
                else:
                    current_q_text.append(t)

    if curr_opt:
        current_options.append(curr_opt)
    flush_question()

    return "\n\n".join(output_blocks)

def parse_with_groq(text_chunk, api_key, model):
    if not GROQ_AVAILABLE:
        raise ImportError("Install groq: pip install groq")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Convert the following questions into ASSIST test format:\n\n{text_chunk}"}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def parse_with_ollama(text_chunk, base_url, model):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Convert the following questions into ASSIST test format:\n\n{text_chunk}"}
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    resp = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()

def process_pdf(uploaded_file, engine_choice, groq_api_key=None, groq_model=None, 
                ollama_url=None, ollama_model=None, progress_callback=None):
    if engine_choice == "Smart Color/Format Parser (Instant & Deterministic)":
        return parse_with_smart_color(uploaded_file)

    with pdfplumber.open(uploaded_file) as pdf:
        full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])

    lines = full_text.split("\n")
    chunks, cur_chunk, cur_len = [], [], 0
    for l in lines:
        cur_chunk.append(l)
        cur_len += len(l)
        if cur_len > 3500:
            chunks.append("\n".join(cur_chunk))
            cur_chunk, cur_len = [], 0
    if cur_chunk:
        chunks.append("\n".join(cur_chunk))

    converted_parts, total_chunks = [], len(chunks)
    for idx, chunk in enumerate(chunks):
        if engine_choice == "Groq API (Cloud LLM)":
            if not groq_api_key:
                raise ValueError("Groq API Key is required.")
            res = parse_with_groq(chunk, groq_api_key, groq_model)
        elif engine_choice == "Ollama (Local LLM)":
            res = parse_with_ollama(chunk, ollama_url, ollama_model)
        else:
            raise ValueError(f"Unknown engine: {engine_choice}")

        converted_parts.append(res)
        if progress_callback:
            progress_callback((idx + 1) / total_chunks)

    return "\n\n".join(converted_parts)