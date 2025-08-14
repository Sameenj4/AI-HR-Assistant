import streamlit as st
from sentence_transformers import SentenceTransformer, util
from ollama import Client
import fitz  # PyMuPDF
import docx

# ----------------- Resume Parser ----------------- #
class ResumeParser:
    def __init__(self, file):
        self.file = file

    def extract_text(self):
        if self.file.name.endswith(".pdf"):
            return self._extract_pdf()
        elif self.file.name.endswith(".docx"):
            return self._extract_docx()
        return ""

    def _extract_pdf(self):
        text = ""
        with fitz.open(stream=self.file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    def _extract_docx(self):
        doc = docx.Document(self.file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    def extract_skills(self, resume_text):
        skill_keywords = [
            "python", "java", "sql", "html", "css", "javascript", "power bi",
            "machine learning", "deep learning", "communication", "teamwork",
            "leadership", "problem solving", "pandas", "numpy", "excel", "react"
        ]
        return [s.title() for s in skill_keywords if s in resume_text.lower()]

# ----------------- Question Generator ----------------- #
class QuestionGenerator:
    def __init__(self, skills):
        self.skills = skills
        self.client = Client(host="http://localhost:11434")

    def generate_questions_and_reference_answers(self):
        prompt = f"Generate HR-style interview questions and their ideal answers for the following skills: {', '.join(self.skills)}. Format each as:\n\nQ: <question>\nA: <ideal answer>\n\nOnly include one question per skill."
        response = self.client.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}]
        )

        lines = response["message"]["content"].strip().split("\n")
        questions = []
        reference_answers = []

        for i in range(0, len(lines), 2):
            if lines[i].startswith("Q:") and i + 1 < len(lines) and lines[i + 1].startswith("A:"):
                question = lines[i][2:].strip()
                answer = lines[i + 1][2:].strip()
                questions.append(question)
                reference_answers.append(answer)

        return questions, reference_answers

# ----------------- Answer Evaluator ----------------- #
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_answer(user_answer, reference):
    model = load_model()
    e1 = model.encode(user_answer, convert_to_tensor=True)
    e2 = model.encode(reference, convert_to_tensor=True)
    score = util.pytorch_cos_sim(e1, e2)
    return round(float(score.item()), 2)

# ----------------- Streamlit App ----------------- #
def main():
    st.set_page_config(page_title="AI HR Assistant", layout="wide")
    st.title("🤖 AI HR Interview Assistant")
    st.write("Upload your resume to get personalized HR questions and AI-based answer evaluation.")

    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
        st.session_state.questions = []
        st.session_state.ref_answers = []
        st.session_state.skills = []
        st.session_state.file = None

    if not st.session_state.interview_started:
        uploaded_file = st.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
        if st.button("🎯 Start Interview") and uploaded_file:
            st.session_state.file = uploaded_file

            with st.spinner("📄 Extracting skills..."):
                parser = ResumeParser(uploaded_file)
                resume_text = parser.extract_text()
                skills = parser.extract_skills(resume_text)

            if not skills:
                st.error("No recognizable skills found.")
                return

            st.session_state.skills = skills
            with st.spinner("🤖 Generating questions and ideal answers using Phi3 (faster model)..."):
                qgen = QuestionGenerator(skills)
                questions, ref_answers = qgen.generate_questions_and_reference_answers()

            st.session_state.questions = questions
            st.session_state.ref_answers = ref_answers
            st.session_state.interview_started = True
            st.rerun()

    else:
        st.success(f"✅ Skills Extracted: {', '.join(st.session_state.skills)}")
        scores = []

        for i, question in enumerate(st.session_state.questions, 1):
            st.subheader(f"Q{i}: {question}")
            key = f"answer_{i}"
            user_answer = st.text_area("✍️ Your Answer:", key=key)

            if user_answer and len(user_answer.strip()) > 3:
                ref = st.session_state.ref_answers[i - 1]
                score = evaluate_answer(user_answer, ref)
                st.markdown(f"🧠 **Similarity Score:** `{score}/1.0`")
                scores.append(score)

                if score >= 0.85:
                    st.success("✅ Great answer!")
                elif score >= 0.65:
                    st.info("👍 Good answer, could be more specific.")
                else:
                    st.warning("⚠️ Try to include a clearer example or more details.")

        if scores:
            avg = round(sum(scores) / len(scores), 2)
            st.markdown(f"### 🏁 Final Interview Score: `{avg}/1.0`")

        st.button("🔄 Restart Interview", on_click=lambda: st.session_state.clear())

if __name__ == "__main__":
    main()
