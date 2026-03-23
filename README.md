
AI-Based Paper Evaluation System

1. Overview

This project is an **AI-powered automated answer sheet evaluation system** built using modern NLP and AI techniques. It evaluates student answers by comparing them with model answers and provides **scores + detailed feedback**.

The system supports both **text input and PDF uploads**, making it practical for real-world academic use.


2. Key Features

* 🔐 User Authentication (Student, Instructor, Admin)
* 📄 PDF-based Answer Upload (OCR using Gemini AI)
* 🧠 Semantic Similarity Evaluation using SBERT
* 📊 Automatic Score Calculation
* 💬 AI-generated Feedback for Students
* 📁 MongoDB Database Integration
* 🌐 Interactive UI using Streamlit
* 📈 Instructor Dashboard with Analytics

---

3. Tech Stack

 Frontend

	Streamlit

Backend

	Python

AI / ML

* Google Gemini API (OCR + Feedback)
* Sentence Transformers (SBERT)

Database

	MongoDB Atlas

System Architecture

1. Student uploads answer (text/PDF)
2. PDF → Text using Gemini OCR
3. Compare with model answer using SBERT
4. Generate similarity score
5. Convert score → marks
6. Generate feedback using AI
7. Store results in MongoDB
8. Display results on dashboard

Project Structure


AI-Paper-Evaluation-System/
│
├── app.py                 # Main Streamlit Application
├── requirements.txt      # Dependencies
├── README.md             # Project Documentation
└── database/             # MongoDB collections
```

---

Installation & Setup

1. Clone the Repository

```bash
git clone https://github.com/jayanth-thiraviam/AI-BASED-PAPER-EVALUATION-SYSTEM/upload
cd AI-Paper-Evaluation-System
```

---

2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

3. Set Environment Variables

```bash
export GEMINI_API_KEY="your_api_key"
```

---

4. Run the Application

```bash
streamlit run app.py
```

---

How It Works

Step 1: Upload Answer Key (Instructor)
* Upload model answers (text/PDF)

Step 2: Student Submission
* Submit answers via text or PDF

Step 3: Evaluation

* SBERT calculates similarity score
* Score converted into marks
* AI generates feedback

Step 4: Results

* Students view scores + feedback
* Instructors view analytics

---

Evaluation Method

* Uses **Cosine Similarity**
* Model: `paraphrase-MiniLM-L6-v2`
* Score Range: 0 → 1
* Converted to marks (e.g., /10)

---

Security Features

* Password hashing using SHA-256
* Role-based access control
* Secure MongoDB connection

---

Future Improvements

* ✅ JWT Authentication
* 📊 Advanced Analytics Dashboard
* 🌐 Deployment (AWS / GCP)
* 🧠 Improved NLP models
* 📱 Mobile-friendly UI

---

Author

JAYANTH THIRAVIAM M

---
