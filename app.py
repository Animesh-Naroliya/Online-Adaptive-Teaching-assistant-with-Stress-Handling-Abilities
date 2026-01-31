import time 
import os
from datetime import datetime
from flask import Flask, jsonify, request, session, render_template, g
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from database import db, User, Conversation, Message
from groqChatbot import llm_chatbot 
from video_analysis.video_analysis import analyze_video_frame
import pyttsx3
import threading

# Set this environment variable for local testing with HTTP
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Load environment variables
load_dotenv()

# Create the Flask app instance
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a-default-secret-key')

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database with the app.py
db.init_app(app)

# Wrap Flask app with SocketIO - CORS enabled
# Using eventlet for asynchronous support for real-time video/audio streams
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Function to create database tables
def create_db():
    with app.app_context():
        db.create_all()
        print("Database tables created!")

def get_current_user():
    user_id = session.get('user_id')
    if user_id:
        return User.query.get(user_id)
    return None

@app.before_request
def load_user():
    g.user = get_current_user()

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Authentication required'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# --- CORE ROUTES ---
# Rendering index.html
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_session')
def check_session():
    user = get_current_user()
    if user:
        return jsonify({
            'is_authenticated': True, 
            'username': user.username
        }), 200
    return jsonify({'is_authenticated': False}), 200

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if not all([name, username, email, password, confirm_password]):
        return jsonify({'success': False, 'message': 'Missing fields'}), 400
    if password != confirm_password:
        return jsonify({'success': False, 'message': 'Passwords do not match'}), 400

    existing_user_email = User.query.filter_by(email=email).first()
    existing_user_username = User.query.filter_by(username=username).first()
    
    if existing_user_email or existing_user_username:
        return jsonify({'success': False, 'message': 'Email or username already registered'}), 409

    new_user = User(name=name, username=username, email=email, password=password)
    db.session.add(new_user)
    db.session.commit()
    session['user_id'] = new_user.id

    return jsonify({'success': True, 'message': 'User created successfully'}), 201

@app.route('/api/profile', methods=['PUT'])
@login_required
def update_profile():
    data = request.get_json()
    likes = data.get('likes')
    dislikes = data.get('dislikes')
    context = data.get('context')

    user = g.user 
    if likes is not None:
        user.likes = ','.join(likes) if isinstance(likes, list) else likes
    if dislikes is not None:
        user.dislikes = ','.join(dislikes) if isinstance(dislikes, list) else dislikes
    if context is not None:
        user.context = context

    db.session.commit()

    return jsonify({'success': True, 'message': 'Profile updated successfully'}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    identifier = data.get('identifier')
    password = data.get('password')

    if not all([identifier, password]):
        return jsonify({'success': False, 'message': 'Missing fields'}), 400

    user = User.query.filter((User.email == identifier) | (User.username == identifier)).first()

    if user and user.check_password(password):
        session['user_id'] = user.id
        return jsonify({'success': True, 'message': 'Login successful'}), 200
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

# CHATBOT & SESSION API 
@app.route('/api/chat', methods=['POST'])
@login_required
def chat_message():
    data = request.get_json()
    message_content = data.get('message')
    conversation_id = data.get('conversation_id')
    emotion_detected = data.get('emotion_detected') 

    if not all([message_content, conversation_id]):
        return jsonify({'success': False, 'message': 'Missing message or conversation ID'}), 400

    conversation = Conversation.query.filter_by(id=conversation_id, user_id=g.user.id).first()

    if not conversation:
        return jsonify({'success': False, 'message': 'Conversation not found'}), 404

    user_message = Message(
        conversation_id=conversation_id,
        sender='user',
        content=message_content,
        emotion_detected=emotion_detected
    )
    db.session.add(user_message)
    db.session.commit()

    # Call LLM API 
    context_text = g.user.context if g.user.context else "a student"
    likes_text = g.user.likes if g.user.likes else ""
    session_topic = conversation.topic if conversation.topic else "general learning"
    llm_response_content = None 

    user_data = {
        'username': g.user.username,
        'context': context_text,
        'likes': likes_text,
        'facial_emotion': emotion_detected,
        'session_topic': session_topic
    }

    try:
        llm_response_content = llm_chatbot.get_response(
            conversation_id, 
            message_content, 
            user_data
        )
    except Exception as e:
        print(f"Global Chatbot Execution Failed: {e}. Falling back to generic response.")
        llm_response_content = None 

    # START OF LLM FALLBACK LOGIC
    if not llm_response_content or llm_response_content.isspace() or 'I apologize, ' in llm_response_content:
        llm_response_content = (
            f"It seems like we're experiencing a technical issue. Don't worry, let's try to resolve this together. The error message is indicating a problem with the LLM API configuration or connectivity. I'm here to help you navigate through any challenges that come up. How would you like to proceed?"
        )
    
    # Save VTA Response
    vta_message = Message(
        conversation_id=conversation_id,
        sender='vta',
        content=llm_response_content
    )
    db.session.add(vta_message)
    db.session.commit()

    return jsonify({
        'success': True, 
        'vta_response': llm_response_content,
        'message_id': vta_message.id
    }), 200

@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    user = g.user
    likes_list = user.likes.split(',') if user.likes else []
    dislikes_list = user.dislikes.split(',') if user.dislikes else []

    return jsonify({
        'success': True,
        'user_id': user.id,
        'name': user.name,
        'username': user.username,
        'email': user.email,
        'likes': likes_list,
        'dislikes': dislikes_list,
        'context': user.context,
    }), 200

@app.route('/api/sessions', methods=['GET'])
@login_required
def get_sessions():
    sessions = Conversation.query.filter_by(user_id=g.user.id).order_by(Conversation.created_at.desc()).limit(10).all()
    
    session_list = [{
        'id': s.id,
        'title': s.title,
        'created_at': s.created_at.strftime("%Y-%m-%d %H:%M")
    } for s in sessions]

    return jsonify({'success': True, 'sessions': session_list}), 200

@app.route('/api/sessions/<int:session_id>/set-topic', methods=['POST'])
@login_required
def set_session_topic(session_id):
    data = request.get_json()
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({'success': False, 'message': 'Topic cannot be empty'}), 400
    
    conversation = Conversation.query.filter_by(id=session_id, user_id=g.user.id).first()
    
    if not conversation:
        return jsonify({'success': False, 'message': 'Conversation not found'}), 404
    
    # Update conversation topic and title
    conversation.topic = topic
    conversation.title = f"{topic[:50]}..." if len(topic) > 50 else topic
    db.session.commit()
    
    # Send acknowledgment message
    ack_message = Message(
        conversation_id=session_id,
        sender='vta',
        content=f"Perfect! Let's dive into **{topic}**. I'm here to help you learn and understand this topic thoroughly. What would you like to know first?"
    )
    db.session.add(ack_message)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'topic': topic,
        'title': conversation.title,
        'acknowledgment': ack_message.content
    }), 200

@app.route('/api/sessions/new', methods=['POST'])
@login_required
def new_session():
    # Create session with auto-generated title
    title = f"Session - {datetime.now().strftime('%b %d, %H:%M')}"
    
    new_conversation = Conversation(
        user_id=g.user.id,
        title=title,
        topic=None  # Topic will be set after user responds
    )
    db.session.add(new_conversation)
    db.session.commit()
    
    # Send combined welcome and topic question message
    welcome_message = Message(
        conversation_id=new_conversation.id,
        sender='vta',
        content=f"Hello {g.user.username}! 👋 Welcome to your new learning session. I'm your Emotion-Aware Virtual Teaching Assistant.\n\nWhat would you like to study today? Please tell me the topic or subject you want to focus on in this session."
    )
    db.session.add(welcome_message)
    db.session.commit()

    return jsonify({
        'success': True, 
        'conversation_id': new_conversation.id,
        'title': new_conversation.title,
        'welcome_message': welcome_message.content,
        'topic_set': False
    }), 201

@app.route('/api/sessions/<int:session_id>/messages', methods=['GET'])
@login_required
def get_session_messages(session_id):
    conversation = Conversation.query.filter_by(id=session_id, user_id=g.user.id).first()

    if not conversation:
        return jsonify({'success': False, 'message': 'Conversation not found'}), 404
        
    messages = Message.query.filter_by(conversation_id=session_id).order_by(Message.timestamp.asc()).all()
    
    message_list = [{
        'id': m.id,
        'sender': m.sender,
        'content': m.content,
        'emotion': m.emotion_detected,
        'timestamp': m.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    } for m in messages]

    return jsonify({
        'success': True, 
        'messages': message_list, 
        'title': conversation.title,
        'topic': conversation.topic,
        'topic_set': conversation.topic is not None
    }), 200

# Text-to-Speech Endpoint with pyttsx3
# Global TTS engine and control
tts_engine = None
tts_lock = threading.Lock()
is_speaking = False

def init_tts_engine():
    global tts_engine
    if tts_engine is None:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)  # Speed
        tts_engine.setProperty('volume', 1.0)  # Volume

@app.route('/api/tts/speak', methods=['POST'])
@login_required
def text_to_speech_speak():
    global is_speaking
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'success': False, 'message': 'No text provided'}), 400
        
        # Initialize engine if needed
        init_tts_engine()
        
        # Start speaking in a thread
        def speak():
            global is_speaking
            with tts_lock:
                is_speaking = True
                try:
                    tts_engine.say(text)
                    tts_engine.runAndWait()
                except:
                    pass
                finally:
                    is_speaking = False
        
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/tts/stop', methods=['POST'])
@login_required
def text_to_speech_stop():
    global is_speaking, tts_engine
    try:
        if is_speaking and tts_engine:
            # Force stop the engine
            try:
                tts_engine.stop()
            except:
                pass
            # Reinitialize engine for next use
            tts_engine = None
            is_speaking = False
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"TTS Stop Error: {str(e)}")
        is_speaking = False
        return jsonify({'success': False, 'message': str(e)}), 500

# SOCKETIO (Real-Time Emotion Detection) 
@socketio.on('video_stream')
def handle_video_stream(data):
    base64_frame = data.get('frame')
    
    if base64_frame:
        detected_emotion = analyze_video_frame(base64_frame)
    else:
        detected_emotion = 'Neutral'
        
    emit('video_response', {'emotion': detected_emotion})


if __name__ == '__main__':
    create_db()
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)