import streamlit as st
import tempfile
import os
import json
from extract import extract

st.set_page_config(page_title="Resume Extractor", layout="wide")
st.title("Resume Extractor")
st.caption("Upload a resume PDF and extract structured JSON using Claude or GPT.")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload resume PDF", type=["pdf"])
    model = st.selectbox("Model", ["gpt", "claude"])
    confidence = st.checkbox("Show confidence scores", value=True)
    run = st.button("Extract", type="primary")

if run and uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner(f"Extracting with {model}..."):
        try:
            result = extract(tmp_path, model, confidence=confidence)
            os.unlink(tmp_path)

            with col2:
                st.subheader("Extracted Data")
                data = result["data"]

                st.markdown(f"**Name:** {data['name']}")
                st.markdown(f"**Location:** {data.get('location', 'N/A')}")
                st.markdown(f"**Email:** {', '.join(data.get('emails', []))}")
                st.markdown(f"**Phone:** {', '.join(data.get('phones', []))}")
                st.markdown(f"**Years Experience:** {data.get('total_years_experience', 'N/A')}")

                if data.get("employment"):
                    st.subheader("Employment")
                    for job in data["employment"]:
                        st.markdown(f"**{job['title']}** at {job['company']} ({job.get('start')} – {job.get('end', 'Present')})")
                        st.caption(job.get("summary", ""))

                if data.get("education"):
                    st.subheader("Education")
                    for edu in data["education"]:
                        st.markdown(f"**{edu['degree']}** — {edu['institution']} ({edu.get('year', '')})")

                if data.get("skills"):
                    st.subheader("Skills")
                    st.write(", ".join(data["skills"]))

                if data.get("certifications"):
                    st.subheader("Certifications")
                    for c in data["certifications"]:
                        st.markdown(f"- {c}")

                if confidence and "confidence" in result:
                    st.subheader("Confidence Scores")
                    for field, score in result["confidence"].items():
                        st.progress(score, text=f"{field}: {score:.0%}")

                st.subheader("Raw JSON")
                st.json(result["data"])

                st.caption(f"Model: {result['model']} | Latency: {result['latency']:.2f}s | Tokens in: {result['in']} out: {result['out']}")

        except Exception as e:
            os.unlink(tmp_path)
            st.error(f"Error: {e}")
elif run and not uploaded:
    st.warning("Please upload a PDF first.")