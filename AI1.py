# -*- coding: utf-8 -*-
"""
Created on Sat May 17 21:32:59 2025

@author: peaco
"""

import streamlit as st
import PyPDF2
import requests

# Use your API key securely from Streamlit secrets
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Query OpenRouter LLM
def query_llm(prompt, model="mistralai/mixtral-8x7b-instruct"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an academic research assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, json=payload, headers=headers)
    result = response.json()
    if "choices" in result:
        return result['choices'][0]['message']['content']
    else:
        raise Exception(f"LLM API error: {result}")

# Prompt builder
def build_prompt(text, style, language, length):
    style_instruction = {
        "Detailed": "Provide a comprehensive and detailed summary.",
        "Bullet Points": "Summarize the content in clear bullet points.",
        "Key Findings": "List only the key findings and conclusions."
    }[style]

    length_instruction = {
        "Brief": "Keep the summary concise and under 100 words.",
        "Medium": "Make the summary about 200 words.",
        "Full": "Do not limit the length — cover all important aspects."
    }[length]

    return (
        f"Summarize the following academic paper. {style_instruction} "
        f"{length_instruction} Write the output in {language}.\n\n{text}"
    )

# Chunk text to fit LLM limits
def chunk_text(text, max_chunk_size=4000, overlap=500):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_chunk_size - overlap
    return chunks

# Streamlit UI
st.set_page_config(page_title="AI PDF Summarizer with Chunking", layout="centered")
st.title("📄 AI Research Paper Summarizer with Chunking")

st.markdown("Upload a PDF and get a detailed summary using a free LLM via OpenRouter.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

language = st.selectbox("🔠 Choose output language", [
    "English", "Spanish", "French", "German", "Hindi", "Chinese", "Arabic"
])

style = st.selectbox("🧾 Choose summarization style", [
    "Detailed", "Bullet Points", "Key Findings"
])

length = st.radio("📏 Choose summary length", ["Brief", "Medium", "Full"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    if len(text) > 20000:
        st.warning(f"Your document is quite long ({len(text)} characters). It will be summarized in chunks for best results.")

    if st.button("🧠 Summarize"):
        with st.spinner("Summarizing in chunks..."):
            chunks = chunk_text(text)
            chunk_summaries = []

            for i, chunk in enumerate(chunks, 1):
                st.info(f"Processing chunk {i} of {len(chunks)}")
                prompt = build_prompt(chunk, style, language, length)
                try:
                    summary = query_llm(prompt)
                    chunk_summaries.append(summary)
                except Exception as e:
                    st.error(f"Chunk {i} failed: {e}")
                    chunk_summaries.append("❌ Error generating summary for this chunk.")

            # Combine chunk summaries
            combined_summary_text = "\n\n".join(chunk_summaries)

            # Optionally, summarize the combined summary if length is brief or medium
            if length != "Full":
                with st.spinner("Generating final combined summary..."):
                    final_prompt = (
                        f"Summarize the following combined summaries "
                        f"into a {length.lower()} summary in {language}:\n\n{combined_summary_text}"
                    )
                    try:
                        final_summary = query_llm(final_prompt)
                        st.subheader(f"📝 Final Summary ({language}, {style}, {length})")
                        st.write(final_summary)
                    except Exception as e:
                        st.error(f"Failed to generate final summary: {e}")
                        st.write(combined_summary_text)
            else:
                st.subheader(f"📝 Combined Chunk Summaries ({language}, {style}, {length})")
                st.write(combined_summary_text)


# --- CV Entry Example ---
st.markdown("---")
st.markdown("### CV Entry Example")
st.markdown("""
- Developed an AI-powered PDF summarization web app using Streamlit and OpenRouter’s free LLM API.
- Implemented chunk-based text processing to handle large academic papers efficiently.
- Incorporated multi-language support and customizable summarization styles and lengths.
- Deployed the app on Streamlit Cloud with secure API key management.
""")
