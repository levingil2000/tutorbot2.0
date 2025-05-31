from flask import Flask, request, jsonify, render_template, redirect, url_for
import uuid
import datetime
import os
import google.generativeai as genai
import dotenv

# Load environment variables
dotenv.load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Mock databases
lessons = {}  # lesson_id: {topic, objectives, assessment, teacher_instructions, chat_history}
sessions = {}  # session_token: {lesson_id, student_chat, score, rating, duration}

@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/teacher')
def teacher_dashboard():
    return render_template('teacher.html')

@app.route('/student')
def student_dashboard():
    return render_template('student.html')

@app.route('/api/create_lesson', methods=['POST'])
def create_lesson():
    data = request.json
    topic = data['topic']
    lesson_id = str(uuid.uuid4())
    lessons[lesson_id] = {
        'topic': topic,
        'objectives': [],
        'assessment': [],
        'teacher_instructions': [],
        'chat_history': [],
    }
    return jsonify({'lesson_id': lesson_id})

@app.route('/api/update_lesson', methods=['POST'])
def update_lesson():
    data = request.json
    lesson_id = data['lesson_id']
    lessons[lesson_id]['objectives'] = data['objectives']
    lessons[lesson_id]['assessment'] = data['assessment']
    lessons[lesson_id]['teacher_instructions'].append(data['instructions'])
    return jsonify({'message': 'Lesson updated'})

@app.route('/api/finalize_lesson', methods=['POST'])
def finalize_lesson():
    lesson_id = request.json['lesson_id']
    session_token = str(uuid.uuid4())
    sessions[session_token] = {
        'lesson_id': lesson_id,
        'student_chat': [],
        'score': None,
        'rating': None,
        'start_time': datetime.datetime.now(),
        'end_time': None
    }
    return jsonify({'access_token': session_token})

@app.route('/api/student_chat', methods=['POST'])
def student_chat():
    data = request.json
    token = data['access_token']
    message = data['message']
    sessions[token]['student_chat'].append({'from': 'student', 'message': message})

    # Generate response using Gemini
    try:
        chat = model.start_chat(history=[])
        response = chat.send_message(message).text
    except Exception as e:
        response = f"Error contacting Gemini API: {e}"

    sessions[token]['student_chat'].append({'from': 'ai', 'message': response})
    return jsonify({'response': response})

@app.route('/api/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.json
    token = data['access_token']
    score = data['score']
    sessions[token]['score'] = score
    return jsonify({'message': 'Score recorded'})

@app.route('/api/submit_rating', methods=['POST'])
def submit_rating():
    data = request.json
    token = data['access_token']
    rating = data['rating']
    sessions[token]['rating'] = rating
    sessions[token]['end_time'] = datetime.datetime.now()
    return jsonify({'message': 'Rating recorded'})

@app.route('/api/analytics/<lesson_id>')
def analytics(lesson_id):
    results = []
    for token, session in sessions.items():
        if session['lesson_id'] == lesson_id:
            duration = (session['end_time'] - session['start_time']).total_seconds() / 60 if session['end_time'] else None
            results.append({
                'session_token': token,
                'score': session['score'],
                'rating': session['rating'],
                'duration': duration,
                'evaluation': f"AI handled topic '{lessons[lesson_id]['topic']}' with overall interaction of {len(session['student_chat'])} messages."
            })
    return jsonify(results)

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
