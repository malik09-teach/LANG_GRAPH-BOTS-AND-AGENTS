import ollama

# System prompt defining strict non-punctuation rules
SYSTEM_INSTRUCTION = """
You are an expert medical exam formatting agent. Your sole job is to clean up and structure text multiple-choice questions into a raw plaintext import format.

Strict Formatting Rules:
1. Start every question stem line with a single '?' symbol.
2. Prefix the verified correct answer choice line with a '+' symbol.
3. Prefix all incorrect distractor options with a '-' symbol.
4. Remove any existing choice labels like 'A.', 'B.', '1.', '#', or labels like 'A)'.
5. ABSOLUTELY DO NOT INCLUDE trailing punctuation signs at the end of question lines or option lines. Strip out trailing periods (.), commas (,), colons (:), and semicolons (;).
6. Output ONLY the raw formatted questions. Do not include any conversation, introductions, notes, meta-text, markdown bold headers, or explanations. 

Example Target Format:
?How many stages have the development of surgery in Tajikistan
- 2
- 3
+ 5
- 4
- 1
"""

def clean_trailing_punctuation(text):
    """
    Python fallback loop to cleanly strip trailing punctuation line by line.
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped_line = line.rstrip()
        # Keep stripping any trailing punctuation characters from the end
        while stripped_line and stripped_line[-1] in ['.', ',', ':', ';']:
            stripped_line = stripped_line[:-1].rstrip()
        cleaned_lines.append(stripped_line)
    return "\n".join(cleaned_lines)

def process_chunk_with_ollama(text_chunk, model_name="gemma2:2b"):
    """
    Sends a small slice of text to local Ollama with strict parsing parameters.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"Convert the following text chunk:\n\n{text_chunk}"}
            ],
            options={
                "temperature": 0.0  # Force strict adherence to formatting rules
            }
        )
        raw_llm_output = response['message']['content']
        
        # Apply the fallback clean function to guarantee zero trailing punctuation symbols
        return clean_trailing_punctuation(raw_llm_output)
        
    except Exception as e:
        return f"\n[ERROR PROCESSING CHUNK: {str(e)}]\n"
