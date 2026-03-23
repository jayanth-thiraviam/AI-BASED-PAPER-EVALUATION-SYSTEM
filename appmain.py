import streamlit as st
st.set_page_config(
        page_title="AI Based Paper Evaluation System",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
import os
import time
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import base64
from tempfile import NamedTemporaryFile
import pandas as pd
import json
import hashlib
import uuid
import datetime
from streamlit_cookies_manager import CookieManager


import google.generativeai as genai
import os

def configure_gemini():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyC54lXELzfgZQU-VOc8rIdgSHG52CZDwrk"))

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    ocr_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",  
        generation_config=generation_config,
    )

    feedback_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
    )

    return ocr_model, feedback_model

def connect_to_mongodb():
    client = MongoClient("mongodb+srv://thiraviam07070:Care20032004@cluster0.yqffp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

    db = client["answer_evaluation_system"] 
    return db

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            time.sleep(2)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    return True

def extract_text_from_pdf(ocr_model, pdf_file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_file_path = temp_file.name

    try:
        gemini_file = upload_to_gemini(temp_file_path, mime_type="application/pdf")
        wait_for_files_active([gemini_file])
        
        response = ocr_model.generate_content([gemini_file])
        extracted_text = "".join(part.text for part in response.parts if hasattr(part, "text"))

        return extracted_text

    finally:
        os.unlink(temp_file_path)

def evaluate_with_sbert(student_answer, model_answer):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    student_embedding = model.encode(student_answer, convert_to_tensor=True)
    model_embedding = model.encode(model_answer, convert_to_tensor=True)
    
    similarity_score = util.pytorch_cos_sim(student_embedding, model_embedding).item()
    
    return similarity_score

def generate_feedback(feedback_model, student_answer, model_answer, similarity_score):
    
    feedback = {
        "strengths": [],
        "missing_concepts": [],
        "improvements": [],
        "encouragement": ""
    }
    
    if similarity_score > 0.8:
        feedback["strengths"].append("Your answer closely aligns with the model answer, demonstrating a strong grasp of the key concepts.")
        feedback["strengths"].append("Your response is well-structured and effectively communicates the key points.")
    elif similarity_score > 0.5:
        feedback["strengths"].append("You have captured several important ideas but could further refine your response.")
        feedback["strengths"].append("Your understanding is developing well; consider elaborating on key concepts.")
    else:
        feedback["strengths"].append("Your answer includes some relevant points, but there is significant room for improvement.")
        feedback["strengths"].append("You show an effort in addressing the question, which is a great starting point.")
    
    if similarity_score < 1.0:
        feedback["missing_concepts"].append("Consider incorporating more specific details from the model answer, such as key definitions, examples, or explanations.")
    if similarity_score < 0.7:
        feedback["missing_concepts"].append("Your response could benefit from a clearer explanation of fundamental concepts.")
    if similarity_score < 0.4:
        feedback["missing_concepts"].append("It may help to review the underlying principles behind the question and ensure they are reflected in your answer.")
    
    feedback["improvements"].append("Try to structure your answer more clearly, ensuring that all key points are logically connected.")
    if similarity_score < 0.6:
        feedback["improvements"].append("Consider breaking down complex ideas into smaller, more digestible points to improve clarity.")
    if similarity_score < 0.4:
        feedback["improvements"].append("Revisit the model answer carefully and analyze how it addresses each part of the question.")
    if similarity_score < 0.2:
        feedback["improvements"].append("It may help to study additional resources or examples to gain a deeper understanding of the topic.")
    
    if similarity_score > 0.7:
        feedback["encouragement"] = "You're doing great! Keep refining your responses and aiming for completeness."
    elif similarity_score > 0.4:
        feedback["encouragement"] = "Good effort! With a bit more focus on key concepts, your answers will improve significantly."
    else:
        feedback["encouragement"] = "Don't be discouraged! Learning takes time, and each attempt helps build your understanding. Keep practicing, and you'll get there."
    
    return feedback

def calculate_score(similarity_score, max_marks=10):
    raw_score = similarity_score * max_marks
    rounded_score = round(raw_score * 2) / 2
    return rounded_score

def save_evaluation_results(db, student_id, question_id, student_answer, model_answer, 
                           similarity_score, numerical_score, feedback):

    evaluation_collection = db["evaluations"]
    
    result = {
        "student_id": student_id,
        "question_id": question_id,
        "timestamp": time.time(),
        "student_answer": student_answer,
        "model_answer": model_answer,
        "similarity_score": similarity_score,
        "numerical_score": numerical_score,
        "feedback": feedback
    }
    
    return evaluation_collection.insert_one(result).inserted_id

def register_user(db, username, password, email, role):
    users_collection = db["users"]
    
    if users_collection.find_one({"username": username}):
        return False, "Username already exists"
    
    if users_collection.find_one({"email": email}):
        return False, "Email already exists"
    
    user = {
        "username": username,
        "password": hash_password(password),
        "email": email,
        "role": role,
        "created_at": datetime.datetime.now()
    }
    
    result = users_collection.insert_one(user)
    if result.inserted_id:
        return True, "Registration successful"
    else:
        return False, "Registration failed"

def authenticate_user(db, username, password):
    users_collection = db["users"]
    user = users_collection.find_one({"username": username})
    
    if user and user["password"] == hash_password(password):
        return True, user
    else:
        return False, None

def init_session_state():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Login"

def login_page(db):
    st.title("Login to AI Based Paper Evaluation System")
    
    tab1, tab2 = st.tabs(["Login","Signup"])
    
    with tab1:
        st.subheader("Login")
        with st.form(key="login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(label="Login",use_container_width=True)
            
            if submit_button:
                if username and password:
                    success, user = authenticate_user(db, username, password)
                    if success:
                        st.session_state.user = user
                        st.session_state.authenticated = True
                        st.session_state.current_page = "Dashboard"
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    with tab2:
        st.subheader("Register")
        with st.form(key="register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email = st.text_input("Email")
            role = st.selectbox("Role", ["student", "instructor", "admin"])
            
            submit_button = st.form_submit_button(label="Register",use_container_width=True)
            
            if submit_button:
                if new_username and new_password and confirm_password and email:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        success, message = register_user(db, new_username, new_password, email, role)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.error("Please fill all fields")

def dashboard_page(db, ocr_model, feedback_model):
    st.sidebar.title("AI Based Paper Evaluation System")


    user_role = st.session_state.user.get("role", "student")
    username = st.session_state.user.get("username", "User")
    
    st.title(f"Welcome, {username}!")
    
    if user_role == "admin":
        admin_dashboard(db)
    elif user_role == "instructor":
        instructor_dashboard(db, ocr_model, feedback_model)
    else:
        student_dashboard(db,ocr_model)
    display_sidebar_info()
    
def admin_dashboard(db):
    st.header("Admin Dashboard")
    
    st.subheader("User Management")
    users = list(db["users"].find({}, {"password": 0}))
    
    if users:
        user_df = pd.DataFrame(users)
        st.dataframe(user_df)
    else:
        st.info("No users found in the database.")
    
    st.subheader("System Statistics")
    eval_count = db["evaluations"].count_documents({})
    answer_key_count = db["answer_keys"].count_documents({})
    user_count = db["users"].count_documents({})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", user_count)
    with col2:
        st.metric("Answer Keys", answer_key_count)
    with col3:
        st.metric("Evaluations", eval_count)

def instructor_dashboard(db, ocr_model, feedback_model):
    st.sidebar.title("Instructor Navigation")
    instructor_page = st.sidebar.radio("Go to", ["Upload Answer Keys", "Evaluate Answers", "View Results"])
    
    refresh_interval = st.sidebar.selectbox(
        "Auto-refresh interval",
        [None, 30, 60, 120, 300],
        format_func=lambda x: "Off" if x is None else f"{x} seconds",
        index=0
    )
    
    if refresh_interval is not None:
        st.sidebar.write(f"Auto-refreshing every {refresh_interval} seconds")
        refresh_placeholder = st.sidebar.empty()
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        time_since_refresh = time.time() - st.session_state.last_refresh
        refresh_placeholder.progress(min(time_since_refresh / refresh_interval, 1.0))
        
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = time.time()
            st.rerun()
    
    if st.sidebar.button("Refresh Now",use_container_width=True):
        st.session_state.last_refresh = time.time()
        st.rerun()
    
    if instructor_page == "Upload Answer Keys":
        instructor_upload_keys(db, ocr_model)
    elif instructor_page == "Evaluate Answers":
        evaluation_section(db, ocr_model, feedback_model)
    else:
        instructor_view_results(db)

def student_dashboard(db, ocr_model):
    st.sidebar.title("Student Navigation")
    student_page = st.sidebar.radio("Go to", ["My Results", "Submit Answers"])
    
    student_id = st.session_state.user.get("username")
    
    refresh_interval = st.sidebar.selectbox(
        "Auto-refresh interval",
        [None, 30, 60, 120, 300],
        format_func=lambda x: "Off" if x is None else f"{x} seconds",
        index=0
    )
    
    if refresh_interval is not None:
        st.sidebar.write(f"Auto-refreshing every {refresh_interval} seconds")
        refresh_placeholder = st.sidebar.empty()
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        time_since_refresh = time.time() - st.session_state.last_refresh
        refresh_placeholder.progress(min(time_since_refresh / refresh_interval, 1.0))
        
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = time.time()
            st.rerun()
    
    if st.sidebar.button("Refresh Now",use_container_width=True):
        st.session_state.last_refresh = time.time()
        st.rerun()
    
    if student_page == "My Results":
        student_results_page(db, student_id)
    else:
        student_submit_answers(db, ocr_model)

def instructor_upload_keys(db, ocr_model):
    st.header("Upload Answer Keys")
    
    tab1, tab2 = st.tabs(["Multiple Text Input", "PDF Upload"])
    
    with tab1:
        with st.form(key="answer_key_form_text"):
            subject = st.text_input("Subject")
            exam_id = st.text_input("Exam ID")
            
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=1)

            questions_data = []
            for i in range(num_questions):
                st.subheader(f"Question {i+1}")
                question_id = st.text_input(f"Question ID", key=f"qid_{i}")
                question_text = st.text_input(f"Question Text (optional)", key=f"qtext_{i}")
                model_answer = st.text_area(f"Model Answer", key=f"ans_{i}")
                max_marks = st.number_input(f"Maximum Marks", min_value=1, value=10, key=f"marks_{i}")
                
                questions_data.append({
                    "question_id": question_id,
                    "question_text": question_text,
                    "model_answer": model_answer,
                    "max_marks": max_marks
                })
            
            submit_button = st.form_submit_button(label="Save All Answer Keys",use_container_width=True)
            
            if submit_button:
                if not all([subject, exam_id]):
                    st.error("Please enter Subject and Exam ID")
                    return
                
                valid_questions = []
                for i, q in enumerate(questions_data):
                    if not q["question_id"] or not q["model_answer"]:
                        st.error(f"Question {i+1} is missing Question ID or Model Answer")
                    else:
                        valid_questions.append(q)
                
                if not valid_questions:
                    st.error("No valid questions to save")
                    return
                
                answer_keys_collection = db["answer_keys"]
                inserted_count = 0
                
                for q in valid_questions:
                    result = answer_keys_collection.insert_one({
                        "subject": subject,
                        "exam_id": exam_id,
                        "question_id": q["question_id"],
                        "question_text": q["question_text"],
                        "model_answer": q["model_answer"],
                        "max_marks": q["max_marks"],
                        "created_at": time.time(),
                        "created_by": st.session_state.user.get("username")
                    })
                    
                    if result.inserted_id:
                        inserted_count += 1
                
                if inserted_count > 0:
                    st.success(f"Successfully saved {inserted_count} answer key(s)!")
                else:
                    st.error("Failed to save answer keys.")
    
    with tab2:
        with st.form(key="answer_key_form_pdf"):
            subject_pdf = st.text_input("Subject")
            exam_id_pdf = st.text_input("Exam ID")
            answer_key_file = st.file_uploader("Upload Answer Key (PDF)", type=["pdf"])
            
            st.subheader("PDF Processing Options")
            process_type = st.radio(
                "How should the PDF be processed?",
                ["Extract entire PDF as one answer key", "Split PDF into multiple answer keys"]
            )
            
            if process_type == "Split PDF into multiple answer keys":
                st.info("The system will try to identify questions and answers in the PDF automatically.")
                prefix = st.text_input("Question ID Prefix (e.g., 'Q' will create Q1, Q2, etc.)", value="Q")
                default_marks = st.number_input("Default Maximum Marks per question", min_value=1, value=10)
            elif process_type == "Extract entire PDF as one answer key":
                question_id_pdf = st.text_input("Question ID")
                max_marks_pdf = st.number_input("Maximum Marks", min_value=1, value=10)
            
            submit_pdf_button = st.form_submit_button(label="Process PDF and Save",use_container_width=True)
            
            if submit_pdf_button:
                if not all([subject_pdf, exam_id_pdf, answer_key_file]):
                    st.error("Please fill Subject, Exam ID and upload a PDF")
                    return
                
                if process_type == "Extract entire PDF as one answer key" and not question_id_pdf:
                    st.error("Please enter Question ID")
                    return
                
                with st.spinner("Extracting text from PDF..."):
                    model_answer_pdf = extract_text_from_pdf(ocr_model, answer_key_file)
                
                if model_answer_pdf:
                    answer_keys_collection = db["answer_keys"]
                    
                    if process_type == "Extract entire PDF as one answer key":
                        st.subheader("Extracted Model Answer")
                        st.write(model_answer_pdf)
                        
                        result = answer_keys_collection.insert_one({
                            "subject": subject_pdf,
                            "exam_id": exam_id_pdf,
                            "question_id": question_id_pdf,
                            "model_answer": model_answer_pdf,
                            "max_marks": max_marks_pdf,
                            "created_at": time.time(),
                            "created_by": st.session_state.user.get("username"),
                            "source": "pdf"
                        })
                        
                        if result.inserted_id:
                            st.success(f"Answer key from PDF saved successfully! ID: {result.inserted_id}")
                        else:
                            st.error("Failed to save answer key.")
                    else:
                        import re
                        import streamlit as st
                        import genai

                        def extract_questions_answers(text):
                            qa_pattern = re.compile(r'(?:Q\d+\.|Question\s*\d*:?)(.*?)(?:A\d+\.|Answer\s*\d*:?)(.*?)(?=Q\d+\.|Question\s*\d*:|$)', re.DOTALL | re.IGNORECASE)
                            
                            matches = qa_pattern.findall(text)
                            
                            qa_pairs = []
                            for i, (question, answer) in enumerate(matches, 1):
                                qa_pairs.append(f"QUESTION {i}:\n{question.strip()}\n\nANSWER {i}:\n{answer.strip()}\n")
                            
                            return "\n".join(qa_pairs)

                        with st.spinner("Extracting questions and answers..."):
                            split_text = extract_questions_answers(model_answer_pdf)

                            st.subheader("Extracted Questions and Answers")
                            st.text_area("Separated Q&A", split_text, height=300)
                            
                            qa_pairs = []
                            current_question = ""
                            current_answer = ""
                            question_mode = False
                            answer_mode = False
                            
                            for line in split_text.split('\n'):
                                if line.strip().startswith("QUESTION"):
                                    if current_question and current_answer:
                                        qa_pairs.append({
                                            "question": current_question.strip(),
                                            "answer": current_answer.strip()
                                        })

                                    current_question = line.split(':', 1)[1] if ':' in line else ""
                                    question_mode = True
                                    answer_mode = False
                                elif line.strip().startswith("ANSWER"):
                                    question_mode = False
                                    answer_mode = True
                                    current_answer = line.split(':', 1)[1] if ':' in line else ""
                                elif question_mode:
                                    current_question += "\n" + line
                                elif answer_mode:
                                    current_answer += "\n" + line
                            
                            if current_question and current_answer:
                                qa_pairs.append({
                                    "question": current_question.strip(),
                                    "answer": current_answer.strip()
                                })
                            
                            st.subheader(f"Found {len(qa_pairs)} question-answer pairs")
                            
                            inserted_count = 0
                            for i, qa in enumerate(qa_pairs):
                                question_id = f"{prefix}{i+1}"
                                result = answer_keys_collection.insert_one({
                                    "subject": subject_pdf,
                                    "exam_id": exam_id_pdf,
                                    "question_id": question_id,
                                    "question_text": qa["question"],
                                    "model_answer": qa["answer"],
                                    "max_marks": default_marks,
                                    "created_at": time.time(),
                                    "created_by": st.session_state.user.get("username"),
                                    "source": "pdf_split"
                                })
                                
                                if result.inserted_id:
                                    inserted_count += 1
                            
                            if inserted_count > 0:
                                st.success(f"Successfully saved {inserted_count} answer keys from the PDF!")
                            else:
                                st.error("Failed to save answer keys from the PDF.")
                else:
                    st.error("Failed to extract text from the uploaded PDF.")

def instructor_view_results(db):
    st.header("Evaluation Results")
    
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    answer_keys = db["answer_keys"].distinct("exam_id")
    
    if not answer_keys:
        st.info("No exams found.")
        return
    
    total_evaluations = db["evaluations"].count_documents({})
    recent_evaluations = db["evaluations"].count_documents({"timestamp": {"$gt": time.time() - 86400}})  # Last 24 hours
    pending_submissions = db["submissions"].count_documents({"status": "pending"})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Evaluations", total_evaluations)
    with col2:
        st.metric("Evaluations (24h)", recent_evaluations)
    with col3:
        st.metric("Pending Submissions", pending_submissions, delta=pending_submissions)
    
    selected_exam = st.selectbox("Select Exam", answer_keys)
    
    if selected_exam:
        questions = list(db["answer_keys"].find({"exam_id": selected_exam}, {"question_id": 1}))
        question_ids = [q["question_id"] for q in questions]
        
        evaluations = list(db["evaluations"].find({"question_id": {"$in": question_ids}}))
        
        if not evaluations:
            st.info(f"No evaluations found for exam {selected_exam}")
            
            pending = db["submissions"].count_documents({
                "exam_id": selected_exam,
                "status": "pending"
            })
            
            if pending > 0:
                st.warning(f"There are {pending} pending submissions for this exam waiting to be evaluated.")
                if st.button("Evaluate Pending Submissions",use_container_width=True):
                    st.session_state.evaluate_pending = True
                    st.session_state.pending_exam = selected_exam
                    st.rerun()
            
            return
        
        df = pd.DataFrame(evaluations)
        avg_score = df["numerical_score"].mean()
        max_score = df["numerical_score"].max()
        min_score = df["numerical_score"].min()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Score", f"{avg_score:.2f}")
        with col2:
            st.metric("Maximum Score", f"{max_score:.2f}")
        with col3:
            st.metric("Minimum Score", f"{min_score:.2f}")
        
        st.subheader("Student Performance by Question")
        question_perf = df.groupby("question_id")["numerical_score"].agg(["mean", "min", "max", "count"])

        question_perf_formatted = question_perf.copy()
        question_perf_formatted["mean"] = question_perf_formatted["mean"].round(2)
        question_perf_formatted["min"] = question_perf_formatted["min"].round(2)
        question_perf_formatted["max"] = question_perf_formatted["max"].round(2)
        question_perf_formatted.columns = ["Average", "Minimum", "Maximum", "Count"]
        st.dataframe(question_perf_formatted)
        
        st.subheader("Individual Student Performance")
        student_perf = df.groupby("student_id")["numerical_score"].agg(["mean", "min", "max", "count"])

        student_perf_formatted = student_perf.copy()
        student_perf_formatted["mean"] = student_perf_formatted["mean"].round(2)
        student_perf_formatted["min"] = student_perf_formatted["min"].round(2)
        student_perf_formatted["max"] = student_perf_formatted["max"].round(2)
        student_perf_formatted.columns = ["Average", "Minimum", "Maximum", "Count"]
        st.dataframe(student_perf_formatted)
        
        csv = student_perf_formatted.to_csv().encode('utf-8')
        st.download_button(
            label="Download Student Results as CSV",
            data=csv,
            file_name=f'student_results_{selected_exam}.csv',
            mime='text/csv',use_container_width=True
        )
        
        st.subheader("Detailed Results")

        filter_student = st.text_input("Filter by Student ID", "")
        filter_question = st.selectbox("Filter by Question", ["All"] + question_ids)
        
        filtered_evaluations = evaluations
        if filter_student:
            filtered_evaluations = [e for e in filtered_evaluations if filter_student.lower() in e['student_id'].lower()]
        if filter_question != "All":
            filtered_evaluations = [e for e in filtered_evaluations if e['question_id'] == filter_question]
        
        st.write(f"Showing {len(filtered_evaluations)} of {len(evaluations)} evaluations")
        
        for evaluation in filtered_evaluations:
            with st.expander(f"Student: {evaluation['student_id']} - Question: {evaluation['question_id']} - Score: {evaluation['numerical_score']:.2f}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Student Answer:**")
                    st.write(evaluation["student_answer"])
                
                with col2:
                    st.write("**Model Answer:**")
                    st.write(evaluation["model_answer"])
                
                st.write("**Feedback:**")
                st.write(evaluation["feedback"])
                
                eval_time = datetime.datetime.fromtimestamp(evaluation.get("timestamp", 0))
                st.caption(f"Evaluated on: {eval_time.strftime('%Y-%m-%d %H:%M:%S')}")

def student_results_page(db, student_id):
    st.header("My Evaluation Results")
    
    evaluations = list(db["evaluations"].find({"student_id": student_id}))
    
    if not evaluations:
        st.info("No evaluations found. Your answers haven't been evaluated yet.")
        return
    
    answer_keys_collection = db["answer_keys"]
    
    question_ids = [eval["question_id"] for eval in evaluations]
    answer_keys = list(answer_keys_collection.find({"question_id": {"$in": question_ids}}))
    
    question_to_exam = {key["question_id"]: key["exam_id"] for key in answer_keys}
    
    for eval in evaluations:
        eval["exam_id"] = question_to_exam.get(eval["question_id"], "Unknown")

    exams = list(set([eval["exam_id"] for eval in evaluations]))
    
    for exam in exams:
        st.subheader(f"Exam: {exam}")
        
        exam_evals = [eval for eval in evaluations if eval["exam_id"] == exam]
        
        total_score = sum([eval["numerical_score"] for eval in exam_evals])
        max_possible = sum([answer_keys_collection.find_one({"question_id": eval["question_id"]}).get("max_marks", 10) 
                         for eval in exam_evals])
        
        st.metric("Total Score", f"{total_score:.2f}/{max_possible}")
        
        comprehensive_feedback = None
        try:
            feedback_collection = db.get_collection("comprehensive_feedback")
            feedback_doc = feedback_collection.find_one({
                "student_id": student_id,
                "exam_id": exam
            })
            if feedback_doc:
                comprehensive_feedback = feedback_doc["comprehensive_feedback"]
        except Exception:
            pass
        
        if not comprehensive_feedback and exam_evals:
            comprehensive_feedback = exam_evals[0]["feedback"]
        
        if comprehensive_feedback:
            st.subheader("Comprehensive Feedback")
            st.write(comprehensive_feedback)
        
        st.subheader("Individual Question Results")
        
        for eval in exam_evals:
            with st.expander(f"Question: {eval['question_id']} - Score: {eval['numerical_score']:.2f}"):
                st.write("**Your Answer:**")
                st.write(eval["student_answer"])
                
                st.write(f"**Similarity Score:** {eval['similarity_score']:.2f}")
                
                if st.checkbox(f"Show Model Answer for {eval['question_id']}", key=f"model_{eval['question_id']}"):
                    st.write("**Model Answer:**")
                    st.write(eval["model_answer"])
                    
def student_submit_answers(db, ocr_model):
    st.header("Submit Answers")
    
    answer_keys_collection = db["answer_keys"]
    exams = list(answer_keys_collection.distinct("exam_id"))
    
    if not exams:
        st.warning("No exams available for submission. Please contact your instructor.")
        return
    
    st.subheader("Your Submission Status")
    student_id = st.session_state.user.get("username")
    submissions = list(db["submissions"].find({"student_id": student_id}))
    
    if submissions:
        exam_submissions = {}
        for sub in submissions:
            exam_id = sub.get("exam_id", "Unknown")
            if exam_id not in exam_submissions:
                exam_submissions[exam_id] = []
            exam_submissions[exam_id].append(sub)
        
        for exam_id, subs in exam_submissions.items():
            with st.expander(f"Exam: {exam_id} ({len(subs)} submission(s))"):
                sub_data = []
                for sub in subs:
                    status = sub.get("status", "pending")
                    status_emoji = "⏳" if status == "pending" else "✅"
                    submission_time = datetime.datetime.fromtimestamp(sub.get("submitted_at", 0))
                    
                    sub_data.append({
                        "Question": sub.get("question_id", "Unknown"),
                        "Status": f"{status_emoji} {status.capitalize()}",
                        "Submitted At": submission_time.strftime("%Y-%m-%d %H:%M"),
                        "Source": sub.get("source", "text")
                    })
                
                if sub_data:
                    st.dataframe(pd.DataFrame(sub_data))
    else:
        st.info("You haven't submitted any answers yet.")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Multiple Text Answers", "PDF Upload"])
    
    with tab1:
        with st.form(key="submit_answers_form"):
            exam_id = st.selectbox("Select Exam", options=exams)
            
            questions = list(answer_keys_collection.find({"exam_id": exam_id}))
            
            if not questions:
                st.warning(f"No questions found for exam {exam_id}")
                st.form_submit_button(label="Submit Answers", disabled=True,use_container_width=True)
                return
            
            already_submitted = []
            for q in questions:
                question_id = q["question_id"]
                existing = db["submissions"].find_one({
                    "student_id": student_id,
                    "exam_id": exam_id,
                    "question_id": question_id
                })
                if existing:
                    already_submitted.append(question_id)
            
            if already_submitted:
                st.warning(f"You have already submitted answers for questions: {', '.join(already_submitted)}")
            
            st.write("Please answer the following questions:")
            
            answers = []
            for i, question in enumerate(questions):
                question_id = question["question_id"]
                question_text = question.get("question_text", f"Question {question_id}")
                
                is_submitted = question_id in already_submitted
                status_text = " (already submitted)" if is_submitted else ""
                
                st.subheader(f"Question {i+1}: {question_text}{status_text}")
                answer_text = st.text_area(f"Your Answer for {question_id}", key=f"ans_{i}")
                
                answers.append({
                    "question_id": question_id,
                    "answer": answer_text,
                    "already_submitted": is_submitted
                })
            
            resubmit = st.checkbox("Allow resubmission for already submitted answers")
            submit_button = st.form_submit_button(label="Submit All Answers",use_container_width=True)
            
            if submit_button:
                valid_answers = []
                for ans in answers:
                    if ans["answer"].strip() and (not ans["already_submitted"] or resubmit):
                        valid_answers.append({
                            "question_id": ans["question_id"],
                            "answer": ans["answer"]
                        })
                
                if not valid_answers:
                    st.error("Please provide at least one answer or check the resubmission box")
                    return
                
                submissions_collection = db["submissions"]
                inserted_count = 0
                updated_count = 0
                
                for ans in valid_answers:
                    existing = submissions_collection.find_one({
                        "student_id": student_id,
                        "exam_id": exam_id,
                        "question_id": ans["question_id"]
                    })
                    
                    if existing and resubmit:
                        result = submissions_collection.update_one(
                            {"_id": existing["_id"]},
                            
                                "$set": {
                                    "answer": ans["answer"],
                                    "submitted_at": time.time(),
                                    "status": "pending"  
                                }
                        )
                        if result.modified_count > 0:
                            updated_count += 1
                    elif not existing:
                        result = submissions_collection.insert_one({
                            "student_id": student_id,
                            "exam_id": exam_id,
                            "question_id": ans["question_id"],
                            "answer": ans["answer"],
                            "submitted_at": time.time(),
                            "status": "pending"
                        })
                        
                        if result.inserted_id:
                            inserted_count += 1
                
                if inserted_count > 0 or updated_count > 0:
                    message = f"Successfully submitted {inserted_count} new answer(s)"
                    if updated_count > 0:
                        message += f" and updated {updated_count} existing answer(s)"
                    st.success(message + "!")
                    
                    time.sleep(1)  
                    st.experimental_rerun()
                else:
                    st.error("Failed to submit your answers. Please try again.")
                
    with tab2:
        with st.form(key="submit_pdf_form"):
            exam_id_pdf = st.selectbox("Select Exam", options=exams, key="pdf_exam")
            answer_pdf = st.file_uploader("Upload your answers (PDF)", type=["pdf"])
            
            st.subheader("PDF Processing Options")
            process_type = st.radio(
                "How should your PDF be processed?",
                ["Process entire PDF as one answer", "Extract multiple answers from PDF"]
            )
            
            if process_type == "Process entire PDF as one answer":
                questions_pdf = list(answer_keys_collection.find({"exam_id": exam_id_pdf}))
                question_ids_pdf = [q["question_id"] for q in questions_pdf]
                
                if not question_ids_pdf:
                    st.warning(f"No questions found for exam {exam_id_pdf}")
                    st.form_submit_button(label="Submit PDF", disabled=True,use_container_width=True)
                    return
                
                question_id_pdf = st.selectbox("Select Question", options=question_ids_pdf, key="pdf_question")
            
            submit_pdf_button = st.form_submit_button(label="Process PDF and Submit",use_container_width=True)
            
            if submit_pdf_button:
                if not answer_pdf:
                    st.error("Please upload a PDF")
                    return
                
                with st.spinner("Extracting text from PDF..."):
                    extracted_text = extract_text_from_pdf(ocr_model, answer_pdf)
                
                if extracted_text:
                    st.subheader("Extracted Answer(s)")
                    st.write(extracted_text)
                    
                    submissions_collection = db["submissions"]
                    
                    if process_type == "Process entire PDF as one answer":
                        result = submissions_collection.insert_one({
                            "student_id": st.session_state.user.get("username"),
                            "exam_id": exam_id_pdf,
                            "question_id": question_id_pdf,
                            "answer": extracted_text,
                            "submitted_at": time.time(),
                            "status": "pending",
                            "source": "pdf"
                        })
                        
                        if result.inserted_id:
                            st.success("Your answer has been submitted successfully!")
                        else:
                            st.error("Failed to submit your answer.")
                    else:
                        import re
                        import streamlit as st
                        from pymongo import MongoClient

                        def match_answers_to_questions(questions, extracted_answers):
                           
                            matched_answers = {}
                            
                            for question in questions:
                                q_text = question.get("question_text", f"Question {question['question_id']}")
                                q_id = question["question_id"]
                                
                                matched_answer = next((ans["answer"] for ans in extracted_answers if q_text in ans["question"]), "No answer provided")
                                
                                matched_answers[q_id] = matched_answer
                            
                            return matched_answers

                        client = MongoClient("mongodb://localhost:27017/")
                        db = client.exam_database
                        answer_keys_collection = db.answer_keys

                        st.title("Exam Answer Matcher")

                        with st.spinner("Matching answers to questions..."):
                            all_questions = list(answer_keys_collection.find({"exam_id": exam_id_pdf}))
                            
                            if not all_questions:
                                st.error(f"No questions found for exam {exam_id_pdf}")
                            else:
                                extracted_answers = extract_questions_answers(extracted_text)
                                matched_answers = match_answers_to_questions(all_questions, extracted_answers)
                                
                                for q_id, answer in matched_answers.items():
                                    st.write(f"**Answer to Question {q_id}:**\n{answer}\n")

                            
                            st.subheader("Matched Answers to Questions")
                            st.text_area("Processed Answers", matched_text, height=300)
                            
                            answers = []
                            current_question_idx = -1
                            current_answer = ""
                            answer_mode = False
                            
                            for line in matched_text.split('\n'):
                                if line.strip().startswith("ANSWER TO QUESTION"):
                                    if current_question_idx >= 0 and current_answer:
                                        if current_question_idx < len(question_ids):
                                            if "No answer provided" not in current_answer:
                                                answers.append({
                                                    "question_id": question_ids[current_question_idx],
                                                    "answer": current_answer.strip()
                                                })
                                    
                                    try:
                                        q_num = int(line.strip().split("QUESTION")[1].split(":")[0].strip()) - 1
                                        current_question_idx = q_num
                                        current_answer = line.split(':', 1)[1] if ':' in line else ""
                                    except:
                                        current_question_idx = -1
                                        current_answer = ""
                                    
                                    answer_mode = True
                                elif answer_mode and current_question_idx >= 0:
                                    current_answer += "\n" + line
                            
                            if current_question_idx >= 0 and current_answer and current_question_idx < len(question_ids):
                                if "No answer provided" not in current_answer:
                                    answers.append({
                                        "question_id": question_ids[current_question_idx],
                                        "answer": current_answer.strip()
                                    })
                            
                            inserted_count = 0
                            for ans in answers:
                                result = submissions_collection.insert_one({
                                    "student_id": st.session_state.user.get("username"),
                                    "exam_id": exam_id_pdf,
                                    "question_id": ans["question_id"],
                                    "answer": ans["answer"],
                                    "submitted_at": time.time(),
                                    "status": "pending",
                                    "source": "pdf_extracted"
                                })
                                
                                if result.inserted_id:
                                    inserted_count += 1
                            
                            if inserted_count > 0:
                                st.success(f"Successfully submitted {inserted_count} answer(s) from your PDF!")
                            else:
                                st.error("Could not match any answers to questions in the PDF.")
                else:
                    st.error("Failed to extract text from the uploaded PDF.")           

def evaluation_section(db, ocr_model, feedback_model):
    st.header("Evaluate Answer Sheets")
    
    answer_keys_collection = db["answer_keys"]
    exams = list(answer_keys_collection.distinct("exam_id"))
    
    if not exams:
        st.warning("No exam answer keys found. Please upload answer keys first.")
        return
    
    submissions_collection = db["submissions"]
    
    selected_exam = st.selectbox("Filter by Exam (Optional)", options=["All Exams"] + exams)
    
    filter_query = {"status": "pending"}
    if selected_exam != "All Exams":
        filter_query["exam_id"] = selected_exam
    
    pending_submissions = list(submissions_collection.find(filter_query).sort("submitted_at", 1))
    
    if pending_submissions:
        student_exam_groups = {}
        for submission in pending_submissions:
            student_id = submission["student_id"]
            exam_id = submission["exam_id"]
            key = f"{student_id}_{exam_id}"
            
            if key not in student_exam_groups:
                student_exam_groups[key] = {
                    "student_id": student_id,
                    "exam_id": exam_id,
                    "submissions": []
                }
            
            student_exam_groups[key]["submissions"].append(submission)
        
        st.info(f"There are {len(pending_submissions)} pending student submissions from {len(student_exam_groups)} student-exam combinations waiting for evaluation.")
        
        if st.button("Evaluate All Pending Submissions",use_container_width=True):
            st.session_state.evaluate_pending = True
            st.session_state.evaluate_all = True
        
        student_exam_options = [f"{group['student_id']} - {group['exam_id']}" for key, group in student_exam_groups.items()]
        selected_student_exam = st.selectbox("Or select specific student-exam to evaluate", 
                                            options=["Select..."] + student_exam_options)
        
        if selected_student_exam != "Select...":
            if st.button(f"Evaluate selected: {selected_student_exam}",use_container_width=True):
                st.session_state.evaluate_pending = True
                st.session_state.selected_student_exam = selected_student_exam
                st.session_state.evaluate_all = False
        
        if st.session_state.get("evaluate_pending", False):
            groups_to_evaluate = []
            
            if st.session_state.get("evaluate_all", False):
                groups_to_evaluate = list(student_exam_groups.values())
            elif st.session_state.get("selected_student_exam", None):
                selected = st.session_state.selected_student_exam
                selected_student_id = selected.split(" - ")[0]
                selected_exam_id = selected.split(" - ")[1]
                key = f"{selected_student_id}_{selected_exam_id}"
                
                if key in student_exam_groups:
                    groups_to_evaluate = [student_exam_groups[key]]
            
            for group in groups_to_evaluate:
                student_id = group["student_id"]
                exam_id = group["exam_id"]
                
                with st.expander(f"Evaluating Student: {student_id} - Exam: {exam_id}"):
                    submissions = group["submissions"]
                    
                    submissions.sort(key=lambda x: x["question_id"])
                    
                    all_evaluations = []
                    
                    for submission in submissions:
                        question_id = submission["question_id"]
                        student_answer = submission["answer"]
                        
                        model_answer_doc = answer_keys_collection.find_one({
                            "question_id": question_id
                        })
                        
                        if not model_answer_doc:
                            st.error(f"Model answer not found for question {question_id}")
                            continue
                        
                        model_answer = model_answer_doc["model_answer"]
                        max_marks = model_answer_doc.get("max_marks", 10)
                        
                        similarity_score = evaluate_with_sbert(student_answer, model_answer)
                        numerical_score = calculate_score(similarity_score, max_marks)
                        
                        all_evaluations.append({
                            "question_id": question_id,
                            "student_answer": student_answer,
                            "model_answer": model_answer,
                            "similarity_score": similarity_score,
                            "numerical_score": numerical_score,
                            "max_marks": max_marks
                        })
                        
                        st.write(f"Question {question_id} - Similarity Score: {similarity_score:.2f} - Score: {numerical_score}/{max_marks}")
                    
                    if all_evaluations:
                        total_score = sum(eval_data["numerical_score"] for eval_data in all_evaluations)
                        max_possible = sum(eval_data["max_marks"] for eval_data in all_evaluations)
                        overall_percentage = (total_score / max_possible) * 100 if max_possible > 0 else 0
                        
                        def generate_comprehensive_feedback(student_id, exam_id, total_score, max_possible, overall_percentage, all_evaluations):
                            
                            feedback = [
                                f"Student ID: {student_id}",
                                f"Exam ID: {exam_id}",
                                f"Overall Score: {total_score:.2f}/{max_possible} ({overall_percentage:.2f}%)\n",
                                "### Individual Question Feedback\n"
                            ]
                            
                            for i, eval_data in enumerate(all_evaluations, 1):
                                feedback.append(f"**Question {i} (ID: {eval_data['question_id']}):**")
                                feedback.append(f"- Student's Score: {eval_data['numerical_score']}/{eval_data['max_marks']} (Similarity: {eval_data['similarity_score']:.2f})")
                                feedback.append("**Model Answer:**")
                                feedback.append(eval_data['model_answer'])
                                feedback.append("**Student Answer:**")
                                feedback.append(eval_data['student_answer'])
                                feedback.append("---\n")
                            
                            feedback.append("### Summary and Recommendations\n")
                            feedback.append("- Strengths: Identify strong areas where the student performed well.")
                            feedback.append("- Areas for Improvement: Highlight common mistakes or gaps in understanding.")
                            feedback.append("- Study Recommendations: Provide resources or strategies for improvement.")
                            feedback.append("- Encouragement: Keep it constructive and motivating.")
                            
                            return "\n".join(feedback)

client = MongoClient("mongodb://localhost:27017/")
db = client.exam_database
answer_keys_collection = db.answer_keys

st.title("Exam Answer Matcher & Feedback Generator")

with st.spinner("Matching answers to questions..."):
    all_questions = list(answer_keys_collection.find({"exam_id": exam_id_pdf}))
    
    if not all_questions:
        st.error(f"No questions found for exam {exam_id_pdf}")
    else:
        extracted_answers = extract_questions_answers(extracted_text)
        matched_answers = match_answers_to_questions(all_questions, extracted_answers)
        
        for q_id, answer in matched_answers.items():
            st.write(f"**Answer to Question {q_id}:**\n{answer}\n")

with st.spinner("Generating comprehensive feedback..."):
    feedback_report = generate_comprehensive_feedback(student_id, exam_id, total_score, max_possible, overall_percentage, all_evaluations)
    st.text_area("Comprehensive Feedback:", feedback_report, height=400)

                        
                        with st.spinner("Generating comprehensive exam feedback..."):
                            comprehensive_feedback = feedback_model.generate_content(feedback_prompt).text
                        
                        st.subheader("Comprehensive Exam Feedback")
                        st.write(comprehensive_feedback)
                        
                        for eval_data in all_evaluations:
                            save_evaluation_results(
                                db, student_id, eval_data["question_id"], 
                                eval_data["student_answer"], eval_data["model_answer"],
                                eval_data["similarity_score"], eval_data["numerical_score"], 
                                comprehensive_feedback  
                            )
                            
                            submissions_collection.update_one(
                                {"student_id": student_id, "question_id": eval_data["question_id"], "status": "pending"},
                                {"$set": {"status": "evaluated"}}
                            )
                        
                        feedback_collection = db.get_collection("comprehensive_feedback")

                        st.success(f"Evaluation complete for {student_id} on exam {exam_id}")
    else:
        st.info("No pending submissions to evaluate.")

def logout():
    st.session_state.user = None
    st.session_state.authenticated = False
    st.session_state.current_page = "Login"
    st.rerun()

def display_sidebar_info():

    st.sidebar.info("About")
    st.sidebar.info("""
    This system automatically evaluates student answer sheets using:
    - Semantic similarity matching
    - Constructive feedback
    - Centralized data management
    
    The system processes PDF uploads and provides detailed feedback with accuracy scores.
    """)
    
    st.sidebar.success("How to Use")

    st.sidebar.subheader("For Instructors:")
    st.sidebar.markdown("""
    1. Upload answer keys for your exams
    2. Wait for student submissions
    3. Evaluate submitted answers automatically
    4. View evaluation results and analytics
    """)

    st.sidebar.subheader("For Students:")
    st.sidebar.markdown("""
    1. Upload your answers as PDF or enter text
    2. Wait for instructor evaluation
    3. View your results and feedback
    """)
    
def main():

    init_session_state()
    
    db = connect_to_mongodb()
    
    ocr_model, feedback_model = configure_gemini()
    
    if not st.session_state.authenticated:
        login_page(db)
    else:
        if st.sidebar.button("Logout",use_container_width=True):
            logout()
        
        if st.session_state.current_page == "Dashboard":
            dashboard_page(db, ocr_model, feedback_model)
        elif st.session_state.current_page == "AdminPanel":
            admin_dashboard(db)
        else:
            dashboard_page(db, ocr_model, feedback_model)
        
if __name__ == "__main__":
    main()