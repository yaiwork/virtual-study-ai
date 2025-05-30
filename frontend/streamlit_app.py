# ################    streamlit app  ########################################
import streamlit as st
import requests
import re

# === CONFIG ===
st.set_page_config(page_title="📘 Virtual Study AI Tutor Assistant", page_icon="🤖")
st.title("📘 Virtual Study AI Tutor Assistant")

BACKEND_URL = "http://backend:8000"

# === UTILITIES ===
subject_options = ["Biology", "Physics", "Chemistry", "Mathematics", "English", "Economics"] #"History",'Amharic', "አካባቢ ሳይንስ","ግብረ-ገብ"]
grade_options = [str(i) for i in range(5, 13)]

def normalize_topic(topic: str):
    topic = topic.strip().lower()
    topic = re.sub(r's$', '', topic)
    return topic.capitalize()

def get_chat_as_text(history):
    return "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

def show_util_buttons(state_key):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state[state_key] = []
            st.rerun()
    with col2:
        if st.session_state.get(state_key):
            chat_text = get_chat_as_text(st.session_state[state_key])
            st.download_button(
                label="📥 Download Chat",
                data=chat_text,
                file_name=f"{state_key}_history.txt",
                mime="text/plain")

# === UI NAVIGATION ===
#st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose Your AI Agent:", [
    "Chat",
    "Homework Assistant",
    "Lesson Plan Generator",
    "Exam Questions Generator", # Exam & Quiz Generator
    "Study Guide Generator"])

# === CHAT MODE ===
if mode == "Chat":
    st.header("💬 Ask Any Question ")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            res = requests.post(f"{BACKEND_URL}/chat", json={
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            answer = res.json().get("answer", "⚠️ No response from assistant.")
        except Exception as e:
            answer = f"❌ Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer, unsafe_allow_html=True)

    show_util_buttons("chat_history")


# modified
elif mode == "Homework Assistant":
    st.header("📝 Homework Assistant")

    question = st.text_area("Enter your homework question:")

    uploaded_file = st.file_uploader(
        "📎 Upload homework file (PDF, Word, or Image)",
        type=["pdf", "docx", "png", "jpg", "jpeg"]
    )

    if "homework_assistant_chat" not in st.session_state:
        st.session_state["homework_assistant_chat"] = []

    if st.button("Get Help"):
        if not question and not uploaded_file:
            st.warning("Please enter a question or upload a file.")
        else:
            try:
                data = {"question": question}
                files = {}

                if uploaded_file:
                    file_type = uploaded_file.type
                    file_bytes = uploaded_file.getvalue()
                    files["file"] = (uploaded_file.name, file_bytes, file_type)

                # Always use the same endpoint
                if files:
                    response = requests.post(
                        f"{BACKEND_URL}/homework-assistant",
                        data=data,
                        files=files
                    )
                else:
                    response = requests.post(
                        f"{BACKEND_URL}/homework-assistant",
                        json=data
                    )

                answer = response.json().get("answer", "⚠️ No response.")
                st.markdown("### ✅ Answer:")
                st.markdown(answer, unsafe_allow_html=True)

                # Save to chat history
                st.session_state["homework_assistant_chat"].append({"role": "user", "content": question})
                st.session_state["homework_assistant_chat"].append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Utility buttons: Clear & Download
    show_util_buttons("homework_assistant_chat")


# === LESSON PLAN GENERATOR ===
elif mode == "Lesson Plan Generator":
    st.header("📚 Lesson Plan Generator")
    subject = st.selectbox("Subject", subject_options)
    grade = st.selectbox("Grade", grade_options)
    topic = st.text_input("Topic")

    if "lesson_plan_chat" not in st.session_state:
        st.session_state.lesson_plan_chat = []

    for msg in st.session_state.lesson_plan_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if st.button("Generate Lesson Plan"):
        if not all([subject, grade, topic]):
            st.warning("Please fill all fields.")
        else:
            question = f"Generate lesson plan for {topic} in Grade {grade} {subject}"
            st.session_state.lesson_plan_chat.append({"role": "user", "content": question})
            payload = {"subject": subject, "grade": grade, "topic": normalize_topic(topic)}
            try:
                res = requests.post(f"{BACKEND_URL}/lesson-plan", json=payload)
                plan = res.json().get("plan", "⚠️ No response.")
                st.session_state.lesson_plan_chat.append({"role": "assistant", "content": plan})
                with st.chat_message("assistant"):
                    st.markdown(plan, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    show_util_buttons("lesson_plan_chat")

# === EXAM GENERATOR ===
elif mode == "Exam Questions Generator":
    st.header("🧪 Exam Questions Generator")
    subject = st.selectbox("Subject", subject_options, key="exam_subject")
    grade = st.selectbox("Grade", grade_options, key="exam_grade")
    topic = st.text_input("Topic", key="exam_topic")
    num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)

    if "exam_chat" not in st.session_state:
        st.session_state.exam_chat = []

    for msg in st.session_state.exam_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if st.button("Generate Questions"):
        if not all([subject, grade, topic]):
            st.warning("Please fill all fields.")
        else:
            question = f"Generate {num_questions} exam questions on {topic} (Grade {grade} {subject})"
            st.session_state.exam_chat.append({"role": "user", "content": question})
            payload = {"subject": subject, "grade": grade, "topic": normalize_topic(topic)}
            try:
                res = requests.post(f"{BACKEND_URL}/generate-exam", json=payload)
                questions = res.json().get("questions", "⚠️ No response.")
                st.session_state.exam_chat.append({"role": "assistant", "content": questions})
                with st.chat_message("assistant"):
                    st.markdown(questions, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    show_util_buttons("exam_chat")

# === STUDY GUIDE GENERATOR ===

elif mode == "Study Guide Generator":
    st.header("📖 Study Guide Generator")
    subject = st.selectbox("Subject", subject_options, key="sg_subject")
    grade = st.selectbox("Grade", grade_options, key="sg_grade")
    topic = st.text_input("Topic", key="sg_topic")

    if "study_guide_chat" not in st.session_state:
        st.session_state.study_guide_chat = []

    for msg in st.session_state.study_guide_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if st.button("Generate Study Guide"):
        if not all([subject, grade, topic]):
            st.warning("Please fill all fields.")
        else:
            question = f"Study guide for {topic} in Grade {grade} {subject}"
            st.session_state.study_guide_chat.append({"role": "user", "content": question})
            payload = {"subject": subject, "grade": grade, "topic": normalize_topic(topic)}
            try:
                res = requests.post(f"{BACKEND_URL}/study-guide", json=payload)
                guide = res.json().get("study_guide", "⚠️ No response.")
                st.session_state.study_guide_chat.append({"role": "assistant", "content": guide})
                with st.chat_message("assistant"):
                    st.markdown(guide, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    show_util_buttons("study_guide_chat")



# # === EXAM QUESTIONS SOLVER ===
# elif mode == "Exam Questions Solver":
#     st.header("🧠 Exam Questions Solver")
#     subject = st.selectbox("Subject", subject_options, key="solver_subject")
#     grade = st.selectbox("Grade", grade_options, key="solver_grade")
#     question = st.text_area("Enter the exam question you'd like to solve:")

#     if "exam_solver_chat" not in st.session_state:
#         st.session_state.exam_solver_chat = []

#     for msg in st.session_state.exam_solver_chat:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"], unsafe_allow_html=True)

#     if st.button("Solve Question"):
#         if not all([subject, grade, question.strip()]):
#             st.warning("Please fill all fields.")
#         else:
#             user_prompt = f"Exam question for Grade {grade} {subject}: {question}"
#             st.session_state.exam_solver_chat.append({"role": "user", "content": user_prompt})

#             payload = {"subject": subject, "grade": grade, "question": question}
#             try:
#                 res = requests.post(f"{BACKEND_URL}/solve-exam-question", json=payload)
#                 solution = res.json().get("solution", "⚠️ No response.")
#                 st.session_state.exam_solver_chat.append({"role": "assistant", "content": solution})
#                 with st.chat_message("assistant"):
#                     st.markdown(solution, unsafe_allow_html=True)
#             except Exception as e:
#                 st.error(f"❌ Error: {e}")


