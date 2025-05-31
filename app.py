# app.py
from flask import Flask, render_template, request, session, redirect, url_for
import google.generativeai as genai
import os
import uuid
import json
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key for session security
app.config['SESSION_TYPE'] = 'filesystem'  # Configure session storage type

# Configure Gemini AI with API key from environment
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
# Initialize the Gemini 1.5 Flash model
model = genai.GenerativeModel('gemini-1.5-flash')

# In-memory storage for lessons and sessions (replace with database in production)
lessons = {}  # Stores all created lessons by access token
sessions = {}  # Stores active student sessions by session ID

# System prompts for Gemini AI
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
        # Generate lesson content using Gemini AI
        response = model.generate_content(TEACHER_PROMPT + "\nTopic: " + session['topic'])
        try:
            # Attempt to parse AI response as JSON
            session['lesson_data'] = json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            session['lesson_data'] = {
                "objectives": ["Error in lesson generation. Please provide more specific feedback."],
                "workflow": ["Start with basic concepts"],
                "assessment": [{"question": "Sample question?", "answer": "Sample answer"}],
                "practice_quiz": []
            }
            app.logger.error(f"Failed to parse lesson JSON: {response.text}")
    
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
            # Regenerate lesson plan with teacher feedback
            try:
                response = model.generate_content(
                    f"Modify lesson plan based on: {feedback}\nCurrent plan: {json.dumps(session['lesson_data'])}"
                )
                # Update lesson data with modified version
                session['lesson_data'] = json.loads(response.text)
            except json.JSONDecodeError:
                app.logger.error(f"Failed to parse modified lesson JSON: {response.text}")
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
        
        # Generate context for AI
        context = "\n".join([f"{role}: {msg}" for role, msg in session_data['conversation']])
        
        # Generate tutor response using Gemini
        try:
            response = model.generate_content(
                STUDENT_PROMPT + 
                f"\nLesson Plan: {json.dumps(lesson)}" +
                f"\nCurrent Step: {session_data['current_step']}" +
                f"\n\n{context}"
            )
            tutor_response = response.text
        except Exception as e:
            # Handle AI generation errors
            app.logger.error(f"AI generation error: {str(e)}")
            tutor_response = "I'm having trouble responding right now. Please try again."
        
        # Add tutor response to conversation
        session_data['conversation'].append(("tutor", tutor_response))
        
        # Update progress based on tutor response
        if "assessment completed" in tutor_response.lower():
            # Mark workflow as completed
            session_data['current_step'] = len(lesson['workflow'])
        elif "practice quiz" in tutor_response.lower():
            # Mark as quiz phase
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
    app.run(debug=True)  # Run in debug mode for easier troubleshooting
