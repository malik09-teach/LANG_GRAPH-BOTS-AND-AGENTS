import streamlit as st
from agent import process_chunk_with_ollama

st.set_page_config(page_title="Local MCQ Test Bank Agent", page_icon="🤖", layout="wide")

st.title("🤖 Local MCQ Test Bank Agent (Ollama)")
st.write("Convert large unformatted test docs into clean plain import syntax locally with zero trailing punctuation symbols.")

# Configuration Sidebar
st.sidebar.header("⚙️ Agent Settings")
selected_model = st.sidebar.text_input("Local Ollama Model Name", value="llama3")
lines_per_chunk = st.sidebar.slider("Lines per processing chunk", min_value=10, max_value=100, value=40)

# File Uploader
uploaded_file = st.file_uploader("Upload your raw text file (.txt)", type=["txt"])

if uploaded_file is not None:
    # Read text lines from the uploaded file
    raw_bytes = uploaded_file.read()
    raw_text = raw_bytes.decode("utf-8")
    lines = raw_text.splitlines()
    
    st.info(f"Loaded file with {len(lines)} lines of text.")
    
    if st.button("🚀 Start Conversion Process"):
        final_output = []
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Segment lines into blocks to preserve context limit windows
        chunks = [lines[i:i + lines_per_chunk] for i in range(0, len(lines), lines_per_chunk)]
        total_chunks = len(chunks)
        
        # Real-time processing loop
        for idx, chunk_lines in enumerate(chunks):
            status_text.text(f"Processing chunk {idx + 1} of {total_chunks}...")
            
            chunk_payload = "\n".join(chunk_lines)
            formatted_chunk = process_chunk_with_ollama(chunk_payload, model_name=selected_model)
            
            final_output.append(formatted_chunk)
            
            # Progress tracker update
            progress_bar.progress((idx + 1) / total_chunks)
            
        status_text.success("🎉 Complete! Your full file has been processed by the agent.")
        
        # Combine the finalized list into a clean text block
        compiled_output_text = "\n\n".join(final_output)
        
        # Show text area preview
        st.subheader("📝 Preview of Formatted Content")
        st.text_area("Output Preview", compiled_output_text, height=300)
        
        # Present Download Link Button
        st.download_button(
            label="💾 Download Fully Formatted .txt File",
            data=compiled_output_text,
            file_name="formatted_test_bank_clean.txt",
            mime="text/plain"
        )
