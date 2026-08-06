import os
import re
import st_yled
import streamlit as st
import time
from pypdf import PdfReader
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st_yled.init()

st.sidebar.title("Tools",)
page = st.sidebar.radio("Choose a tool:", ["Chat with your PDF", "Meeting Notes Assistant","AI Resume Reviewer"])


def pdf_chat():
    st_yled.title("Chat with your PDF", color="#F57272FF")
    container = st.container(border=True, width="stretch", autoscroll=True)
    container.write("Upload a PDF and ask questions about its content. The app will extract text, chunk it, and use an LLM to answer your questions based on the document.")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        st.pdf(uploaded_file, height=700)
        # STAGE 1 — extract text from the PDF, then chunk it
        reader = PdfReader(uploaded_file)
        text = ""
        with st.spinner("Extracting text from PDF..."):
            time.sleep(1)  # simulate processing time
            for page in reader.pages:
                text += page.extract_text()

            # Split into ~500-character chunks with 50-char overlap
            text = text.replace("\n", " ")
            chunk_size, overlap = 500, 50
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
            chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

            # STAGE 2 — embed & store (fresh collection each upload)
            db = chromadb.Client()
            try:
                db.delete_collection("pdf_docs")   # clear old data on re-upload
            except:
                pass
            collection = db.create_collection("pdf_docs")
            collection.add(documents=chunks, ids=[f"c{i}" for i in range(len(chunks))])

        st.success(f"Loaded {len(chunks)} chunks. Ask a question below.")

        # STAGE 3 & 4 — retrieve then generate
        question = st.text_input("Your question:")
        if question:
            results = collection.query(query_texts=[question], n_results=5)
            context = "\n".join(results["documents"][0])

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Answer using ONLY the context provided. If it's not in the context, say you don't know."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
            )
            with st.container(border=True):
                st.write(response.choices[0].message.content)

    sentiment_mapping = ["one", "two", "three", "four", "five"]
    selected = st.feedback("stars")
    if selected is not None:
        st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")
    st.badge("Made by ruben(Github:R7brn)", color="blue", width="stretch")


def meeting_notes_assistant():
    st_yled.title("Meeting Notes Assistant", color="#F57272FF")
    container = st.container(border=True, width="stretch", autoscroll=True)
    container.write("Paste in raw, messy meeting notes and ask questions about its content")
    meeting_notes = st.text_area("Paste in raw, messy meeting notes here:", height=200, placeholder="Paste your meeting notes here...")
    if meeting_notes:

        def ask(query, instruction):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"Answer using ONLY the context provided. If it's not in the context, say you don't know. {instruction}"},
                    {"role": "user", "content": f"Context:\n{meeting_notes}\n\nQuestion: {query}"}
                ]
            )
            with st.container(border=True):
                st.write(response.choices[0].message.content)

        if st.button("Summarize into 3 bullets"):
            ask("Summarize the meeting", "Give a clean 3-bullet summary.")

        if st.button("List action items (who does what)"):
            ask("List the action items", "Give a list of action items (who does what).")

        question = st.text_input("Or ask your own question:")
        if st.button("Answer a question") and question:
            ask(question, "Answer the user's question about the meeting notes.")

def ai_resume_reviewer():
    # --- Setup ---
    load_dotenv()
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("🚨 GROQ_API_KEY not found!")
        st.stop()

    # --- Page Config ---
    st_yled.title("AI Resume Reviewer", color="#F57272FF")
    # --- Hero Section ---
    container = st.container(border=True, width="stretch", autoscroll=True)
    container.write("Upload your resume and paste a job description to get an AI-powered analysis of how well your resume matches the job. The AI will provide a match score, highlight strengths and gaps")

    # --- Input Section ---
    st.markdown('<div class="section-title">📥 Step 1 — Provide Your Materials</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📄 Your Resume")
        resume_file = st.file_uploader(
            "Drop your PDF here",
            type=["pdf"],
            label_visibility="collapsed"
        )
        if resume_file:
            st.success(f"✅ {resume_file.name}")

    with col2:
        st.markdown("#### 💼 Job Description")
        job_description = st.text_area(
            "Paste the JD",
            height=180,
            placeholder="Paste the full job description here...",
            label_visibility="collapsed"
        )

    # --- Analyze Button ---
    st.markdown('<div class="section-title">🚀 Step 2 — Analyze</div>', unsafe_allow_html=True)
    analyze = st.button("✨ Analyze My Match", type="primary", use_container_width=True)

    # --- Empty State ---
    if not analyze and not (resume_file and job_description):
        st.info("👆 Upload your resume and paste a job description above to get started!")

    # --- Analysis ---
    if analyze:
        if not resume_file or not job_description:
            st.warning("⚠️ Please upload BOTH a resume and job description.")
        else:
            # Extract text
            reader = PdfReader(resume_file)
            resume_text = "".join([page.extract_text() or "" for page in reader.pages])

            prompt = f"""
            You are an expert career coach and ATS analyzer.
            
            Compare the RESUME against the JOB DESCRIPTION and provide:
            
            1. **Match Score**: A number out of 100 (format exactly as "Score: X/100")
            2. **Top 3 Strengths**: Where the resume aligns well
            3. **Top 3 Gaps**: Missing skills, keywords, or experience
            4. **3 Actionable Suggestions**: Specific improvements
            5. **Missing Keywords**: List 5-10 important keywords from the JD not in the resume
            
            RESUME:
            {resume_text}
            
            JOB DESCRIPTION:
            {job_description}
            
            Use clear markdown headers and bullet points.
            """

            with st.spinner("🧠 AI is analyzing your resume... this takes ~10 seconds"):
                groq_client = Groq(api_key=api_key)
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are an expert career coach and ATS analyzer."},
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.choices[0].message.content

            st.markdown("---")
            st.markdown('<div class="section-title">📊 Your Results</div>', unsafe_allow_html=True)

            # --- Score Display ---
            score_match = re.search(r'(\d{1,3})\s*/\s*100', result)
            if score_match:
                score = int(score_match.group(1))
                
                if score >= 80:
                    color, emoji, verdict = "#10B981", "🟢", "Excellent Match!"
                elif score >= 60:
                    color, emoji, verdict = "#F59E0B", "🟡", "Good Match — Room to Improve"
                else:
                    color, emoji, verdict = "#EF4444", "🔴", "Needs Work — Very Fixable!"
                
                st.markdown(f"""
                <div class="score-container">
                    <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em;">Match Score</p>
                    <p class="score-number" style="color: {color};">{score}<span style="font-size: 2rem; color: #6B7280;">/100</span></p>
                    <p style="font-size: 1.25rem; margin: 0; color: {color};">{emoji} {verdict}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(score / 100)
                st.markdown("<br>", unsafe_allow_html=True)

            # --- Analysis Details ---
            st.markdown('<div class="section-title">🔍 Detailed Analysis</div>', unsafe_allow_html=True)
            container = st.container(border=True, width="stretch", autoscroll=True)
            container.markdown(result)
            
            # --- Actions ---
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    label="📥 Download Report",
                    data=result,
                    file_name="resume_analysis.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_b:
                if st.button("🔄 Analyze Another", use_container_width=True):
                    st.rerun()

if page == "Chat with your PDF":
    pdf_chat()
elif page == "Meeting Notes Assistant":
    meeting_notes_assistant()
else:
    ai_resume_reviewer()
