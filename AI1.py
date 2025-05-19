# AI PDF Summarizer - Human-style code

import streamlit as st
import PyPDF2
import requests

# API key - stored in secrets
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Grab text from a PDF
def get_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    full_text = ''
    for pg in reader.pages:
        content = pg.extract_text()
        if content:
            full_text += content + "\n"
    return full_text

# Talk to the LLM
def call_llm_api(prompt, model_name="mistralai/mixtral-8x7b-instruct"):
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are an academic research assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    resp = requests.post(endpoint, headers=headers, json=payload)
    data = resp.json()

    if "choices" in data:
        return data['choices'][0]['message']['content']
    else:
        raise Exception(f"Error from API: {data}")

# Build the prompt string
def make_prompt(txt, s, lang, ln):
    style_map = {
        "Detailed": "Provide a comprehensive and detailed summary.",
        "Bullet Points": "Summarize the content in clear bullet points.",
        "Key Findings": "List only the key findings and conclusions."
    }

    len_map = {
        "Brief": "Keep the summary concise and under 100 words.",
        "Medium": "Make the summary about 200 words.",
        "Full": "Do not limit the length — cover all important aspects."
    }

    return (
        f"Summarize the following academic paper. {style_map[s]} "
        f"{len_map[ln]} Write the output in {lang}.\n\n{txt}"
    )

# Break big text into smaller chunks
def split_into_chunks(text, max_len=4000, overlap=500):
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunk = text[start:end]
        pieces.append(chunk)
        start += max_len - overlap
    return pieces

# App UI
st.set_page_config(page_title="AI PDF Summarizer", layout="centered")
st.title("AI Research Paper Summarizer")

st.write("Upload a PDF and get a summary using OpenRouter.")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

lang = st.selectbox("Output Language", ["English", "Spanish", "French", "German", "Hindi", "Chinese", "Arabic"])
style = st.selectbox("Summary Style", ["Detailed", "Bullet Points", "Key Findings"])
length = st.radio("Summary Length", ["Brief", "Medium", "Full"])

# Main logic
if uploaded_file is not None:
    with st.spinner("Reading PDF..."):
        text = get_pdf_text(uploaded_file)

    if len(text) > 20000:
        st.warning(f"PDF is long ({len(text)} chars). Using chunking to summarize in parts.")

    if st.button("Summarize"):
        with st.spinner("Working..."):
            chunks = split_into_chunks(text)
            summaries = []

            for idx, part in enumerate(chunks, 1):
                st.write(f"Summarizing chunk {idx}/{len(chunks)}...")
                p = make_prompt(part, style, lang, length)
                try:
                    sm = call_llm_api(p)
                    summaries.append(sm)
                except Exception as err:
                    st.error(f"Problem with chunk {idx}: {err}")
                    summaries.append("Error generating summary for this part.")

            full_summary = "\n\n".join(summaries)

            if length != "Full":
                st.write("Combining all chunk summaries...")
                final_p = f"Please create a {length.lower()} summary in {lang} from the following:\n\n{full_summary}"
                try:
                    final_summary = call_llm_api(final_p)
                    st.subheader(f"Final Summary ({lang})")
                    st.write(final_summary)
                except Exception as err:
                    st.error("Final summarization failed.")
                    st.write(full_summary)
            else:
                st.subheader("Combined Chunk Summaries")
                st.write(full_summary)
