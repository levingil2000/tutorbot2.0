# app.py
from flask import Flask, render_template, request, session, redirect, url_for
import os
import uuid
import json
import datetime
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'

# --- IMPROVED HUGGING FACE CONFIGURATION ---
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it in your .env file.")

# List of reliable free models (in order of preference)
AVAILABLE_MODELS = [
    "microsoft/DialoGPT-medium",
    "facebook/blenderbot-400M-distill",
    "microsoft/DialoGPT-small",
    "google/flan-t5-base",
    "HuggingFaceH4/zephyr-7b-beta"
]

# Function to test model availability
def test_model_availability(model_name, max_retries=2):
    """Test if a model is available and working"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": "Hello, this is a test."},
                timeout=10
            )
            if response.status_code == 200:
                return True
            elif response.status_code == 503:
                # Model is loading, wait and retry
                time.sleep(5)
                continue
        except Exception:
            pass
    return False

# Find the first available model
def get_working_model():
    """Get the first working model from the list"""
    for model in AVAILABLE_MODELS:
        app.logger.info(f"Testing model: {model}")
        if test_model_availability(model):
            app.logger.info(f"Using model: {model}")
            return model
    
    # Fallback to a simple model
    app.logger.warning("No models available, using fallback")
    return "microsoft/DialoGPT-small"

# Get working model
WORKING_MODEL = get_working_model()

# Improved HF client with error handling
class SafeInferenceClient:
    def __init__(self, model_name, token):
        self.model_name = model_name
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    def generate_text(self, prompt, max_tokens=500, temperature=0.7, max_retries=3):
        """Generate text with robust error handling"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '').strip()
                    elif isinstance(result, dict):
                        return result.get('generated_text', '').strip()
                elif response.status_code == 503:
                    # Model is loading
                    app.logger.info(f"Model loading, waiting... (attempt {attempt + 1})")
                    time.sleep(10)
                    continue
                else:
                    app.logger.error(f"API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                app.logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        return "I'm having trouble generating a response. Please try again."

# Initialize clients
hf_client = SafeInferenceClient(WORKING_MODEL, HF_TOKEN)

# In-memory storage for lessons and sessions
lessons = {}
sessions = {}

# Improved prompts for better text generation
TEACHER_PROMPT = """Create a lesson plan for the given topic. Format your response as JSON with these sections:
- objectives: List 3-5 learning goals
- workflow: Step-by-step teaching process
- assessment: 5 questions with answers
- practice_quiz: Practice questions with hints

Topic: """

STUDENT_PROMPT = """You are a helpful AI tutor. Your role is to:
1. Guide students through lessons step by step
2. Check understanding before moving forward
3. Provide clear explanations with examples
4. Be encouraging and supportive
5. Ask questions to test comprehension

Current lesson context: """

def generate_lesson_plan(topic):
    """Generate a lesson plan using HF model"""
    prompt = TEACHER_PROMPT + topic
    
    try:
        response = hf_client.generate_text(prompt, max_tokens=800, temperature=0.7)
        
        # Try to extract JSON from response
        if '{' in response and '}' in response:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except Exception as e:
        app.logger.error(f"Failed to generate lesson plan: {str(e)}")
    
    # Fallback lesson plan
    return {
        "objectives": [
            f"Understand the basic concepts of {topic}",
            f"Learn key terminology related to {topic}",
            f"Apply knowledge of {topic} in practical scenarios"
        ],
        "workflow": [
            "Introduction to the topic",
            "Explain core concepts with examples",
            "Interactive discussion and questions",
            "Practice exercises",
            "Assessment and review"
        ],
        "assessment": [
            {"question": f"What is the main concept of {topic}?", "answer": "Basic definition and explanation"},
            {"question": f"How does {topic} apply in real life?", "answer": "Practical applications"},
            {"question": f"What are the key benefits of understanding {topic}?", "answer": "Learning outcomes"}
        ],
        "practice_quiz": [
            {"question": f"Define {topic} in your own words", "hint": "Think about the main characteristics"},
            {"question": f"Give an example of {topic}", "hint": "Consider real-world applications"}
        ]
    }

def generate_tutor_response(conversation_history, lesson_data, current_step):
    """Generate tutor response using HF model"""
    # Create context from conversation
    context = "\n".join([f"{role.title()}: {msg}" for role, msg in conversation_history[-5:]])  # Last 5 messages
    
    prompt = f"{STUDENT_PROMPT}\n\nLesson objectives: {', '.join(lesson_data['objectives'])}\n\nConversation:\n{context}\n\nTutor:"
    
    response = hf_client.generate_text(prompt, max_tokens=200, temperature=0.8)
    
    if not response or len(response.strip()) < 10:
        return "That's interesting! Can you tell me more about what you're thinking? I'm here to help you learn."
    
    return response

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/teacher', methods=['GET', 'POST'])
def teacher_interface():
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'create_topic':
            # Step 1: User enters topic
            session['topic'] = request.form['topic']
            session['lesson_data'] = None
            return redirect(url_for('lesson_plan'))
        
        elif action == 'modify_lesson':
            # Step 2: Handle lesson modifications
            feedback = request.form.get('feedback', '').strip()
            if feedback and session.get('lesson_data'):
                # Modify lesson based on feedback
                modify_prompt = f"Modify this lesson plan based on feedback: '{feedback}'\n\nCurrent plan: {json.dumps(session['lesson_data'])}\n\nProvide the modified plan in JSON format:"
                
                try:
                    response = hf_client.generate_text(modify_prompt, max_tokens=800)
                    if '{' in response and '}' in response:
                        json_start = response.find('{')
                        json_end = response.rfind('}') + 1
                        json_str = response[json_start:json_end]
                        session['lesson_data'] = json.loads(json_str)
                except Exception as e:
                    app.logger.error(f"Failed to modify lesson: {str(e)}")
            
            return redirect(url_for('lesson_plan'))
        
        elif action == 'finalize_lesson':
            # Step 3: Finalize and create access token
            if session.get('lesson_data'):
                token = str(uuid.uuid4())[:8]  # Shorter token for easier sharing
                lessons[token] = {
                    'lesson_data': session['lesson_data'],
                    'topic': session.get('topic', 'Unknown Topic'),
                    'created_at': datetime.datetime.now(),
                    'sessions': []
                }
                
                # Clear session data
                session.pop('lesson_data', None)
                session.pop('topic', None)
                
                return render_template('teacher.html', 
                                     show_token=True,
                                     token=token,
                                     success_message=f"Lesson plan created successfully! Share this access code with students: {token}")
            else:
                return render_template('teacher.html', 
                                     error="No lesson plan found. Please create a lesson plan first.")
    
    return render_template('teacher.html')

@app.route('/lesson_plan')
def lesson_plan():
    if 'topic' not in session:
        return redirect(url_for('teacher_interface'))
    
    # Generate initial lesson plan if not exists
    if not session.get('lesson_data'):
        app.logger.info(f"Generating lesson plan for: {session['topic']}")
        session['lesson_data'] = generate_lesson_plan(session['topic'])
    
    return render_template('lesson_plan.html', 
                          topic=session['topic'],
                          lesson_data=session['lesson_data'])

@app.route('/student', methods=['GET', 'POST'])
def student_interface():
    if request.method == 'POST':
        token = request.form['token'].strip()
        if token in lessons:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                'token': token,
                'start_time': datetime.datetime.now(),
                'current_step': 0,
                'conversation': [],
                'quiz_responses': [],
                'assessment_score': None,
                'rating': None
            }
            
            lesson_topic = lessons[token].get('topic', 'this topic')
            welcome_msg = f"Hello! I'm your AI tutor. Today we'll learn about {lesson_topic}. Let's start with the learning objectives: {', '.join(lessons[token]['lesson_data']['objectives'])}. Are you ready to begin?"
            sessions[session_id]['conversation'].append(("tutor", welcome_msg))
            session['session_id'] = session_id
            return redirect(url_for('tutor_chat'))
        else:
            return render_template('student.html', error="Invalid access token. Please check and try again.")
    return render_template('student.html')

@app.route('/chat', methods=['GET', 'POST'])
def tutor_chat():
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        return redirect(url_for('student_interface'))
    
    session_data = sessions[session_id]
    token = session_data['token']
    lesson = lessons[token]['lesson_data']
    
    if request.method == 'POST':
        user_input = request.form['message']
        session_data['conversation'].append(("student", user_input))
        
        # Generate tutor response
        tutor_response = generate_tutor_response(
            session_data['conversation'], 
            lesson, 
            session_data['current_step']
        )
        
        session_data['conversation'].append(("tutor", tutor_response))
        sessions[session_id] = session_data
    
    return render_template('chat.html', 
                          conversation=session_data['conversation'],
                          lesson_topic=lessons[token].get('topic', 'Unknown Topic'))

@app.route('/complete', methods=['POST'])
def complete_session():
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        return redirect(url_for('index'))
    
    session_data = sessions[session_id]
    session_data['rating'] = request.form.get('rating', 'Not rated')
    session_data['end_time'] = datetime.datetime.now()
    
    duration_minutes = (session_data['end_time'] - session_data['start_time']).seconds // 60
    
    lessons[session_data['token']]['sessions'].append({
        'session_id': session_id,
        'start_time': session_data['start_time'],
        'end_time': session_data['end_time'],
        'duration': duration_minutes,
        'rating': session_data['rating'],
        'score': session_data.get('assessment_score', 'N/A')
    })
    
    session.pop('session_id', None)
    
    return render_template('completion.html', 
                          score=session_data.get('assessment_score', 'N/A'),
                          rating=session_data['rating'],
                          duration=duration_minutes)

@app.route('/analytics/<token>')
def analytics(token):
    if token not in lessons:
        return "Invalid token", 404
    
    lesson_data = lessons[token]
    analytics_data = []
    
    for sess in lesson_data['sessions']:
        analytics_data.append({
            'session_id': sess['session_id'][:8] + "...",
            'date': sess['start_time'].strftime("%Y-%m-%d %H:%M"),
            'duration': f"{sess['duration']} mins",
            'rating': sess['rating'],
            'score': sess['score']
        })
    
    avg_rating = sum([float(s.get('rating', 0)) for s in lesson_data['sessions'] if s.get('rating', '0').isdigit()]) / max(len(lesson_data['sessions']), 1)
    
    return render_template('analytics.html', 
                          token=token,
                          topic=lesson_data.get('topic', 'Unknown Topic'),
                          sessions=analytics_data,
                          total_sessions=len(analytics_data),
                          avg_rating=round(avg_rating, 1))

if __name__ == '__main__':
    app.run(debug=False)
