# app.py
import os
import streamlit as st
from agent import process_pdf

st.set_page_config(
    page_title="PDF to Assyst / ASSIST Converter Agent",
    page_icon="📋",
    layout="wide"
)

st.title("📋 PDF to Assyst / ASSIST Exam Converter")
st.markdown(
    "Convert exam question PDFs into the **ASSIST / Assyst** format (`?Question`, `+Correct`, `-Incorrect`) "
    "using **Groq**, **Ollama**, or **Smart Color Extraction**."
)

with st.sidebar:
    st.header("⚙️ Agent Settings")
    engine_choice = st.radio(
        "Extraction Engine",
        [
            "Smart Color/Format Parser (Instant & Deterministic)", 
            "Groq API (Cloud LLM)", 
            "Ollama (Local LLM)"
        ],
        index=0,
        help="Smart Color Parser automatically extracts questions with colored answers without using LLM tokens. Groq and Ollama use AI models to format unstructured PDFs."
    )
    
    groq_api_key, groq_model, ollama_url, ollama_model = "", "", "", ""

    if engine_choice == "Groq API (Cloud LLM)":
        if not GROQ_AVAILABLE:
            st.error("Please install groq: `pip install groq`")
        groq_api_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
        groq_model = st.selectbox(
            "Groq Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0
        )
    elif engine_choice == "Ollama (Local LLM)":
        ollama_url = st.text_input("Ollama Server URL", value="http://localhost:11434")
        ollama_model = st.text_input("Ollama Model Name", value="llama3:latest")

    st.markdown("---")
    st.markdown(
        "### 💡 Assyst Tips\n"
        "- Download as **`.qst`** or **`.txt`**.\n"
        "- In the Assyst app, use **Add Test ➔ Paste text** to avoid file picker decompression errors."
    )

uploaded_pdf = st.file_uploader("Upload Exam PDF", type=["pdf"])

if uploaded_pdf is not None:
    st.success(f"Uploaded: **{uploaded_pdf.name}**")
    
    if st.button("🚀 Convert PDF to ASSIST Format", type="primary"):
        progress_bar = None
        if engine_choice != "Smart Color/Format Parser (Instant & Deterministic)":
            progress_bar = st.progress(0)
            
        def update_progress(ratio):
            if progress_bar:
                progress_bar.progress(ratio)

        with st.spinner("Processing PDF and extracting questions..."):
            try:
                final_output = process_pdf(
                    uploaded_file=uploaded_pdf,
                    engine_choice=engine_choice,
                    groq_api_key=groq_api_key,
                    groq_model=groq_model,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    progress_callback=update_progress
                )
                st.session_state["converted_output"] = final_output
                st.success("Conversion complete!")
            except Exception as e:
                st.error(f"Error during conversion: {str(e)}")

if "converted_output" in st.session_state and st.session_state["converted_output"]:
    res_text = st.session_state["converted_output"]
    q_count = sum(1 for line in res_text.split("\n") if line.startswith("?"))
    
    st.subheader(f"Conversion Results ({q_count} questions detected)")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.download_button(
            label="📥 Download .txt",
            data=res_text,
            file_name="questions_assist.txt",
            mime="text/plain"
        )
    with col2:
        st.download_button(
            label="📥 Download .qst (Native Assyst)",
            data=res_text,
            file_name="questions_assist.qst",
            mime="text/plain"
        )
    with col3:
        st.info("💡 **Assyst App**: You can also copy the preview text and use **Paste text** in the app.")

    with st.expander("👁️ Preview Formatted Questions (First 5)", expanded=True):
        preview_blocks = res_text.split("\n\n")[:5]
        st.code("\n\n".join(preview_blocks), language="text")