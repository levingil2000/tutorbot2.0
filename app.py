# app.py
from flask import Flask, render_template, request, session, redirect, url_for
import os
import uuid
import json
import datetime
from dotenv import load_dotenv

from huggingface_hub import InferenceClient # Keep this import

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'

# --- NEW HUGGING FACE CONFIGURATION ---
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it in your .env file.")

# IMPORTANT: Explicitly set the base_url to Hugging Face's standard Inference API
HF_INFERENCE_API_BASE_URL = "https://api-inference.huggingface.co/models"

# --- CHANGE MADE HERE: USING HuggingFaceH4/zephyr-7b-beta ---
# This model is generally very accessible on the public Inference API.
HF_MODEL_TEACHER = "HuggingFaceH4/zephyr-7b-beta"
HF_MODEL_STUDENT = "HuggingFaceH4/zephyr-7b-beta"

# Pass the base_url to ensure it hits the correct endpoint
hf_client_teacher = InferenceClient(model=HF_MODEL_TEACHER, token=HF_TOKEN, base_url=HF_INFERENCE_API_BASE_URL)
hf_client_student = InferenceClient(model=HF_MODEL_STUDENT, token=HF_TOKEN, base_url=HF_INFERENCE_API_BASE_URL)


hf_client_teacher = InferenceClient(model=HF_MODEL_TEACHER, token=HF_TOKEN)
hf_client_student = InferenceClient(model=HF_MODEL_STUDENT, token=HF_TOKEN)
# ---------------------------------------------------


# In-memory storage for lessons and sessions (replace with database in production)
lessons = {}  # Stores all created lessons by access token
sessions = {}  # Stores active student sessions by session ID

# System prompts for Gemini AI (these can remain the same)
# NOTE: For Hugging Face chat models, these system prompts are crucial.
# You'll pass them as part of the 'messages' list in the chat_completion call.
TEACHER_PROMPT = """You're an AI lesson planner. Given a topic, generate:
1. Learning objectives (3-5 bullet points)
2. Teaching workflow (step-by-step)
3. 5 assessment questions with answers
4. Practice quiz with hints

Respond in JSON format: {"objectives":[], "workflow":[], "assessment":[], "practice_quiz":[]}"""

STUDENT_PROMPT = """You're a patient AI tutor. Guide students through lessons:
1. Check understanding before proceeding
2. Explain concepts conversationally
3. Provide practice quizzes with hints
4. Administer assessments
5. Be supportive and encouraging"""

# Route: Home/Landing Page
@app.route('/')
def index():
    """Render the landing page with options for teachers and students"""
    return render_template('index.html')

# Route: Teacher Interface
@app.route('/teacher', methods=['GET', 'POST'])
def teacher_interface():
    """
    Handle teacher interface:
    - GET: Show form to enter topic
    - POST: Process submitted topic and redirect to lesson planning
    """
    if request.method == 'POST':
        # Store topic in session and initialize lesson data
        session['topic'] = request.form['topic']
        session['lesson_data'] = None
        return redirect(url_for('lesson_plan'))
    return render_template('teacher.html')

# Route: Lesson Planning
@app.route('/lesson_plan', methods=['GET', 'POST'])
def lesson_plan():
    """
    Handle lesson planning process:
    - Generate initial lesson plan based on topic
    - Process teacher feedback for modifications
    - Finalize lesson and generate access token
    """
    # Redirect if no topic in session
    if 'topic' not in session:
        return redirect(url_for('teacher_interface'))
    
    # Generate initial lesson plan if not exists
    if not session.get('lesson_data'):
        # --- ORIGINAL GEMINI CALL ---
        # response = model.generate_content(TEACHER_PROMPT + "\nTopic: " + session['topic'])
        # ----------------------------

        # --- NEW HUGGING FACE CALL for Teacher Model ---
        try:
            # We use chat_completion because it handles system prompts well.
            # The prompt is constructed to guide the LLM to produce JSON.
            messages = [
                {"role": "system", "content": TEACHER_PROMPT},
                {"role": "user", "content": f"Generate a lesson plan for the topic: {session['topic']}"}
            ]
            
            hf_response = hf_client_teacher.chat_completion(
                messages=messages,
                max_tokens=1500, # Increased tokens for detailed lesson plans
                temperature=0.7,
                # response_model=None # Don't enforce a specific response model for the raw text
            )
            # Extract text from the response object
            response_text = hf_response.choices[0].message.content
        except Exception as e:
            app.logger.error(f"Hugging Face Teacher AI generation error: {str(e)}")
            response_text = "{\"objectives\":[\"Error generating lesson. Please try again or refine your topic.\"], \"workflow\":[\"Start with basic concepts\"], \"assessment\":[{\"question\": \"Sample question?\", \"answer\": \"Sample answer\"}], \"practice_quiz\":[]}"
        # -----------------------------------------------

        try:
            # Attempt to parse AI response as JSON
            session['lesson_data'] = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            app.logger.error(f"Failed to parse lesson JSON from HF response: {response_text}")
            session['lesson_data'] = {
                "objectives": ["Error in lesson generation. Please provide more specific feedback."],
                "workflow": ["Start with basic concepts"],
                "assessment": [{"question": "Sample question?", "answer": "Sample answer"}],
                "practice_quiz": []
            }
            
    # Process teacher feedback
    if request.method == 'POST':
        feedback = request.form.get('feedback', '')
        
        # Handle lesson finalization
        if feedback.lower() == 'finalize':
            # Generate unique access token
            token = str(uuid.uuid4())
            # Store lesson in memory
            lessons[token] = {
                'lesson_data': session['lesson_data'],
                'created_at': datetime.datetime.now(),
                'sessions': []  # Will store student session data
            }
            # Clear session data
            session.pop('lesson_data', None)
            session.pop('topic', None)
            return render_template('teacher.html', token=token)
        
        # Handle lesson modification request
        if feedback.strip():
            # --- ORIGINAL GEMINI CALL ---
            # response = model.generate_content(
            #     f"Modify lesson plan based on: {feedback}\nCurrent plan: {json.dumps(session['lesson_data'])}"
            # )
            # ----------------------------

            # --- NEW HUGGING FACE CALL for Teacher Model Modification ---
            try:
                # The prompt now includes the modification request and the current lesson.
                messages = [
                    {"role": "system", "content": TEACHER_PROMPT + "\nI need the output in JSON format only."},
                    {"role": "user", "content": f"Modify this lesson plan based on the following feedback: '{feedback}'.\n\nHere is the current plan:\n{json.dumps(session['lesson_data'])}"}
                ]
                hf_response = hf_client_teacher.chat_completion(
                    messages=messages,
                    max_tokens=1500, # Keep enough tokens for the full modified plan
                    temperature=0.7,
                    # response_model=None
                )
                response_text = hf_response.choices[0].message.content
            except Exception as e:
                app.logger.error(f"Hugging Face Teacher AI modification error: {str(e)}")
                response_text = "{\"objectives\":[\"Error modifying lesson. Please try again.\"], \"workflow\":[\"Start with basic concepts\"], \"assessment\":[{\"question\": \"Sample question?\", \"answer\": \"Sample answer\"}], \"practice_quiz\":[]}"
            # -------------------------------------------------------------

            try:
                # Update lesson data with modified version
                session['lesson_data'] = json.loads(response_text)
            except json.JSONDecodeError:
                app.logger.error(f"Failed to parse modified lesson JSON from HF response: {response_text}")
                # Keep previous version if parsing fails

    # Render lesson plan for review
    return render_template('teacher.html', 
                            topic=session['topic'],
                            objectives=session['lesson_data']['objectives'],
                            workflow=session['lesson_data']['workflow'],
                            assessment=session['lesson_data']['assessment'])

# Route: Student Interface
@app.route('/student', methods=['GET', 'POST'])
def student_interface():
    """
    Handle student interface:
    - GET: Show form to enter access token
    - POST: Validate token and initialize learning session
    """
    if request.method == 'POST':
        token = request.form['token']
        # Validate access token
        if token in lessons:
            # Create new session ID
            session_id = str(uuid.uuid4())
            # Initialize session data
            sessions[session_id] = {
                'token': token,
                'start_time': datetime.datetime.now(),  # Track session start time
                'current_step': 0,  # Track progress through lesson workflow
                'conversation': [],  # Store chat history
                'quiz_responses': [],  # Store student quiz responses
                'assessment_score': None,  # Track assessment score
                'rating': None  # Track student rating
            }
            # Add welcome message to conversation
            welcome_msg = f"Welcome! Let's learn about '{lessons[token]['lesson_data']['objectives'][0]}'. Ready to begin?"
            sessions[session_id]['conversation'].append(("tutor", welcome_msg))
            # Store session ID in user session
            session['session_id'] = session_id
            return redirect(url_for('tutor_chat'))
        else:
            # Handle invalid token
            return render_template('student.html', error="Invalid access token. Please check and try again.")
    return render_template('student.html')

# Route: Tutor Chat Interface
@app.route('/chat', methods=['GET', 'POST'])
def tutor_chat():
    """
    Handle the conversational tutoring interface:
    - Display chat history
    - Process student messages
    - Generate AI tutor responses
    - Track lesson progress
    """
    # Retrieve session ID from user session
    session_id = session.get('session_id')
    # Redirect if no active session
    if not session_id or session_id not in sessions:
        return redirect(url_for('student_interface'))
    
    # Get session data
    session_data = sessions[session_id]
    token = session_data['token']
    lesson = lessons[token]['lesson_data']
    
    # Process student message
    if request.method == 'POST':
        user_input = request.form['message']
        # Add student message to conversation history
        session_data['conversation'].append(("student", user_input))
        
        # --- ORIGINAL GEMINI CALL ---
        # context = "\n".join([f"{role}: {msg}" for role, msg in session_data['conversation']])
        # response = model.generate_content(
        #     STUDENT_PROMPT + 
        #     f"\nLesson Plan: {json.dumps(lesson)}" +
        #     f"\nCurrent Step: {session_data['current_step']}" +
        #     f"\n\n{context}"
        # )
        # tutor_response = response.text
        # ----------------------------

        # --- NEW HUGGING FACE CALL for Student Model ---
        try:
            # Construct messages list for chat_completion
            # The system prompt should ideally be the first message.
            # Then, reconstruct the conversation for the LLM.
            messages = [
                {"role": "system", "content": STUDENT_PROMPT +
                                               f"\nLesson Plan: {json.dumps(lesson)}" +
                                               f"\nCurrent Step: {session_data['current_step']}"}
            ]
            for role, msg in session_data['conversation']:
                if role == "student": # Map your 'student' role to 'user' for HF models
                    messages.append({"role": "user", "content": msg})
                else: # Map your 'tutor' role to 'assistant' for HF models
                    messages.append({"role": "assistant", "content": msg})
            
            # The last message from the user is what the model should respond to.
            # No need to add the user_input again if it's already in conversation
            # messages.append({"role": "user", "content": user_input}) # this is handled by the loop above

            hf_response = hf_client_student.chat_completion(
                messages=messages,
                max_tokens=300, # Reasonable length for tutor responses
                temperature=0.8, # Slightly higher temperature for more conversational responses
                top_p=0.9
            )
            tutor_response = hf_response.choices[0].message.content
        except Exception as e:
            app.logger.error(f"Hugging Face Student AI generation error: {str(e)}")
            tutor_response = "I'm having trouble responding right now. Please try again."
        # -----------------------------------------------
        
        # Add tutor response to conversation
        session_data['conversation'].append(("tutor", tutor_response))
        
        # Update progress based on tutor response
        if "assessment completed" in tutor_response.lower():
            # Mark workflow as completed
            session_data['current_step'] = len(lesson['workflow'])
        elif "practice quiz" in tutor_response.lower():
            # Mark as quiz phase (if you have specific logic for this)
            session_data['current_step'] = -1
        
        # Save updated session data
        sessions[session_id] = session_data
    
    # Render chat interface
    return render_template('student.html', 
                            conversation=session_data['conversation'],
                            token=token)

# Route: Session Completion
@app.route('/complete', methods=['POST'])
def complete_session():
    """
    Handle session completion:
    - Record student rating
    - Calculate session duration
    - Store session analytics
    - Clean up session data
    """
    session_id = session.get('session_id')
    if not session_id:
        return redirect(url_for('index'))
    
    session_data = sessions[session_id]
    # Store student rating (1-5)
    session_data['rating'] = request.form['rating']
    # Record end time
    session_data['end_time'] = datetime.datetime.now()
    
    # Calculate duration in minutes
    duration_minutes = (session_data['end_time'] - session_data['start_time']).seconds // 60
    
    # Add to lesson analytics
    lessons[session_data['token']]['sessions'].append({
        'session_id': session_id,
        'start_time': session_data['start_time'],
        'duration': duration_minutes,
        'rating': session_data['rating'],
        'score': session_data.get('assessment_score', 'N/A')  # Use 'N/A' if no score
    })
    
    # Clear session
    session.pop('session_id', None)
    
    # Show completion page
    return render_template('student.html', 
                            completed=True,
                            score=session_data.get('assessment_score', 'N/A'),
                            rating=session_data['rating'])

# Route: Analytics Page
@app.route('/analytics/<token>')
def analytics(token):
    """
    Display analytics for a lesson:
    - Show session duration, scores, ratings
    - Requires valid lesson token
    """
    # Validate token
    if token not in lessons:
        return "Invalid token", 404
    
    lesson_data = lessons[token]
    analytics_data = []
    
    # Prepare data for each session
    for sess in lesson_data['sessions']:
        analytics_data.append({
            'session_id': sess['session_id'][:8] + "...",  # Shorten ID for display
            'date': sess['start_time'].strftime("%Y-%m-%d %H:%M"),
            'duration': f"{sess['duration']} mins",
            'rating': sess['rating'],
            'score': sess['score']
        })
    
    # Render analytics page
    return render_template('analytics.html', 
                            token=token,
                            topic=lesson_data['lesson_data']['objectives'][0],
                            sessions=analytics_data)

# Main entry point
if __name__ == '__main__':
    app.run(debug=False)
