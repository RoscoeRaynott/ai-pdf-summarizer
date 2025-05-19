# -*- coding: utf-8 -*-
"""
Created on Sat May 17 21:32:59 2025

@author: peaco
"""

import streamlit as st
import PyPDF2
import requests
import math
import time

# Break text into manageable chunks
def chunk_text(text, max_words=1200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Summarize large text by chunking and combining results
def summarize_in_chunks(text, style, language, length):
    st.info("Splitting the document into chunks...")
    chunks = chunk_text(text)
    summaries = []
    progress_bar = st.progress(0)

    for i, chunk in enumerate(chunks):
        #Error catch
        st.write(f"Style selected: {style}")
        st.write(f"Length selected: {length}")
        st.write(f"Language selected: {language}")
        prompt = build_prompt(chunk, style, language, length)
        try:
            summary = query_llm(prompt)
            summaries.append(f"**Chunk {i+1} Summary:**\n{summary}")
        except Exception as e:
            summaries.append(f"**Chunk {i+1} Failed:** {str(e)}")
        
        progress_bar.progress((i + 1) / len(chunks))
        time.sleep(1)  # Avoid rate limit

    st.success("✅ Summarization complete!")
    return "\n\n---\n\n".join(summaries)

# Set your OpenRouter API key here
API_KEY = "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"#"YOUR_OPENROUTER_API_KEY"

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
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
    return result['choices'][0]['message']['content']

# Streamlit App UI
st.set_page_config(page_title="AI PDF Summarizer", layout="centered")
st.title("📄 AI Research Paper Summarizer")

st.markdown("Upload a PDF and get a summary using a free LLM via OpenRouter.")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Language selection
language = st.selectbox("🔠 Choose output language", [
    "English", "Spanish", "French", "German", "Hindi", "Chinese", "Arabic"
])

# Style selection
style = st.selectbox("🧾 Choose summarization style", [
    "Detailed", "Bullet Points", "Key Findings"
])

# Length selection
length = st.radio("📏 Choose summary length", ["Brief", "Medium", "Full"])

# Prompt builder
def build_prompt(text, style, language, length):
    style_map = {
    "Detailed": "Provide a comprehensive and detailed summary.",
    "Bullet Points": "Summarize the content in clear bullet points.",
    "Key Findings": "List only the key findings and conclusions."
    }
    style_instruction = style_map.get(style, "Provide a concise summary.")
    
    length_map = {
        "Brief": "Keep the summary concise and under 100 words.",
        "Medium": "Make the summary about 200 words.",
        "Full": "Do not limit the length — cover all important aspects."
    }
    length_instruction = length_map.get(length, "Keep the summary concise.")


    return (
        f"Summarize the following academic paper. {style_instruction} "
        f"{length_instruction} Write the output in {language}.\n\n{text}"
    )

# Run summarization
if uploaded_file is not None:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)
        '''
        if len(text) > 5000:
            st.warning("PDF is long. Truncating to 5000 characters.")
            text = text[:5000]
            '''
    if st.button("🧠 Summarize"):
        with st.spinner("Querying the LLM..."):
            summary = summarize_in_chunks(text, style, language, length)
            st.subheader(f"📝 Summary ({language}, {style}, {length})")
            st.write(summary)


