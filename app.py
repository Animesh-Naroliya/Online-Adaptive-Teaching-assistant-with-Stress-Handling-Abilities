import time 
import os
from flask import Flask, jsonify, request, session, redirect, url_for, render_template, g 
from flask_socketio import SocketIO, emit
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.contrib.github import make_github_blueprint, github
from dotenv import load_dotenv
from database import db, User, Conversation, Message
from werkzeug.security import check_password_hash
from datetime import datetime

# --- EXTERNAL MODULE IMPORTS ---
from groqChatbot import llm_chatbot 
from VoiceAnalysis.speechAnalyzer import analyze_audio_blob 
from video_analysis.video_analysis import analyze_video_frame
# ---

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

# Configure Flask-Dance blueprints
# google_bp = make_google_blueprint(
#     client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID"),
#     client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"),
#     scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"]
# )

# github_bp = make_github_blueprint(
#     client_id=os.environ.get("GITHUB_OAUTH_CLIENT_ID"),
#     client_secret=os.environ.get("GITHUB_OAUTH_CLIENT_SECRET"),
#     scope=["user:email"]
# )

# # Register blueprints with the app
# app.register_blueprint(google_bp, url_prefix="/login")
# app.register_blueprint(github_bp, url_prefix="/login")

# --- Utility Functions ---

def get_current_user():
    """Retrieves the current logged-in user."""
    user_id = session.get('user_id')
    if user_id:
        return User.query.get(user_id)
    return None

def login_required(f):
    """Decorator to ensure user is logged in."""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Authentication required'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# --- Request Hooks ---

@app.before_request
def load_user_and_theme():
    """Loads the current user object and their theme into Flask's global context (g)."""
    g.user = get_current_user()
    if g.user:
        # Load user's preferred theme, default to 'dark' if not set
        g.theme = g.user.theme if g.user.theme in ['dark', 'light'] else 'dark'
    else:
        g.theme = 'dark' # Default theme for logged out users

# --- CORE ROUTES ---

# Rendering index.html
@app.route('/')
def index():
    # Frontend will check session via /check_session and render the correct view
    return render_template('index.html')

# # Unified handler for all social logins
# @app.route("/login/<provider>/authorized")
# def social_login_handler(provider):
#     # ... (Keep existing Flask-Dance logic) ...
#     if provider == 'google':
#         service = google
#     elif provider == 'github':
#         service = github
#     else:
#         return jsonify({"message": "Invalid provider"}), 400

#     if not service.authorized:
#         return redirect(url_for(f"{provider}.login"))

#     # user info from the provider's API
#     if provider == 'google':
#         resp = service.get("/oauth2/v2/userinfo")
#         email = resp.json()["email"]
#         name = resp.json().get("name")
#         username = email.split('@')[0]
#     elif provider == 'github':
#         resp = service.get("/user/emails")
#         email = resp.json()[0]["email"]
#         resp = service.get("/user")
#         username = resp.json()["login"]
#         name = resp.json().get("name", username)

#     # Check for an existing user
#     user = User.query.filter_by(email=email).first()

#     # create a new user if one does not exist
#     if not user:
#         # For social login, we generate a random password and set social_id
#         user = User(name=name, username=username, email=email, password=os.urandom(16).hex(), social_id=f"{provider}_{email}")
#         db.session.add(user)
#         db.session.commit()
    
#     session['user_id'] = user.id
#     return render_template('redirect.html')

# Checking if log-in was successful
@app.route('/check_session')
def check_session():
    user = get_current_user()
    if user:
        return jsonify({
            'is_authenticated': True, 
            'username': user.username,
            'theme': user.theme
        }), 200
    
    return jsonify({'is_authenticated': False, 'theme': g.theme}), 200

# Sign-up handler (Uses the new User model initialization)
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

    # Use the User model constructor which hashes the password
    new_user = User(name=name, username=username, email=email, password=password)
    db.session.add(new_user)
    db.session.commit()

    # Log in the user by setting the session ID after successful signup
    session['user_id'] = new_user.id

    return jsonify({'success': True, 'message': 'User created successfully'}), 201

# for profile updates (including likes, dislikes, context)
@app.route('/api/profile', methods=['PUT'])
@login_required
def update_profile():
    data = request.get_json()
    likes = data.get('likes')
    dislikes = data.get('dislikes')
    context = data.get('context')

    user = g.user # Use user loaded in before_request hook
    
    if likes is not None:
        user.likes = ','.join(likes) if isinstance(likes, list) else likes
    if dislikes is not None:
        user.dislikes = ','.join(dislikes) if isinstance(dislikes, list) else dislikes
    if context is not None:
        user.context = context

    db.session.commit()

    return jsonify({'success': True, 'message': 'Profile updated successfully'}), 200

# Login 
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    identifier = data.get('identifier')
    password = data.get('password')

    if not all([identifier, password]):
        return jsonify({'success': False, 'message': 'Missing fields'}), 400

    # Retrieve user by email or username
    user = User.query.filter((User.email == identifier) | (User.username == identifier)).first()

    # NOTE: The original code in app.py had a bug here. I'm assuming the password was intended to be hashed.
    # The database.py check_password method handles the hash check.
    if user and user.check_password(password):
        session['user_id'] = user.id
        return jsonify({'success': True, 'message': 'Login successful'}), 200
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

# log-out
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

# --- CHATBOT & SESSION API ---

# API to handle chat messages
@app.route('/api/chat', methods=['POST'])
@login_required
def chat_message():
    data = request.get_json()
    message_content = data.get('message')
    conversation_id = data.get('conversation_id')
    emotion_detected = data.get('emotion_detected') # From the emotion box

    if not all([message_content, conversation_id]):
        return jsonify({'success': False, 'message': 'Missing message or conversation ID'}), 400

    conversation = Conversation.query.filter_by(id=conversation_id, user_id=g.user.id).first()

    if not conversation:
        return jsonify({'success': False, 'message': 'Conversation not found'}), 404

    # 1. Save User Message
    user_message = Message(
        conversation_id=conversation_id,
        sender='user',
        content=message_content,
        emotion_detected=emotion_detected
    )
    db.session.add(user_message)
    db.session.commit()

    # 2. Call LLM API (Integrated Groq/LangChain)
    context_text = g.user.context if g.user.context else "a student"
    likes_text = g.user.likes if g.user.likes else ""
    llm_response_content = None # Initialize as None

    user_data = {
        'username': g.user.username,
        'context': context_text,
        'likes': likes_text,
        'voice_emotion': emotion_detected, 
        'facial_emotion': emotion_detected
    }

    try:
        # **CALLING GROQ CHATBOT**
        llm_response_content = llm_chatbot.get_response(
            conversation_id, 
            message_content, 
            user_data
        )
    except Exception as e:
        # This catches global execution errors. Groq API errors are handled internally by the chatbot class.
        print(f"Global Chatbot Execution Failed: {e}. Falling back to generic response.")
        llm_response_content = None 

    # **START OF LLM FALLBACK LOGIC**
    # The VTA should attempt a final response even if the LLM fails.
    if not llm_response_content or llm_response_content.isspace() or 'I apologize, ' in llm_response_content:
        # This is the verbose fallback when the Groq API fails.
        llm_response_content = (
            f"It seems like we're experiencing a technical issue. Don't worry, let's try to resolve this together. The error message is indicating a problem with the LLM API configuration or connectivity. I'm here to help you navigate through any challenges that come up. How would you like to proceed?"
        )
    # **END OF LLM FALLBACK LOGIC**
    
    # 3. Save VTA Response
    vta_message = Message(
        conversation_id=conversation_id,
        sender='vta',
        content=llm_response_content
    )
    db.session.add(vta_message)
    db.session.commit()

    # NOTE: REMOVED time.sleep(0.5) to enable immediate frontend streaming

    return jsonify({
        'success': True, 
        'vta_response': llm_response_content,
        'message_id': vta_message.id
    }), 200

# API to retrieve full user profile data
@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    user = g.user
    
    # We must convert the comma-separated strings back to lists for the frontend
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

# API to get a list of past/recent study sessions
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

# API to start a new study session
@app.route('/api/sessions/new', methods=['POST'])
@login_required
def new_session():
    # Simple title generation. Frontend can also send a title
    title = f"New Session - {datetime.now().strftime('%b %d, %H:%M')}"
    
    new_conversation = Conversation(
        user_id=g.user.id,
        title=title
    )
    db.session.add(new_conversation)
    db.session.commit()
    
    # Add a welcoming VTA message to start the thread
    welcome_message = Message(
        conversation_id=new_conversation.id,
        sender='vta',
        content="Welcome! I'm your Emotion-Aware VTA. Let's start a new learning session. How are you feeling today?"
    )
    db.session.add(welcome_message)
    db.session.commit()

    return jsonify({
        'success': True, 
        'conversation_id': new_conversation.id,
        'title': new_conversation.title,
        'welcome_message': welcome_message.content
    }), 201

# API to get messages for a specific session
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

    return jsonify({'success': True, 'messages': message_list, 'title': conversation.title}), 200


# --- SOCKETIO (Real-Time Emotion Detection) ---

# For Video-based emotion-detection module (Facial Recognition)
@socketio.on('video_stream')
def handle_video_stream(data):
    base64_frame = data.get('frame')
    
    if base64_frame:
        # Call the external analysis function
        detected_emotion = analyze_video_frame(base64_frame)
    else:
        detected_emotion = 'Neutral'
        
    # Emit the real-time emotion back to the client
    emit('video_response', {'emotion': detected_emotion})


# For Voice-based emotion-detection module (Acoustic/Speech Recognition)
@socketio.on('audio_stream')
def handle_audio_stream(data):
    # Data sent from the frontend is the binary audio Blob
    audio_blob = data.get('audio') 
    
    results = {}
    
    # 1. Use the SER analyzer (Adapted from user's analyze_audio_blob logic)
    try:
        # **CALLING EXTERNAL SPEECH ANALYZER MODULE**
        results = analyze_audio_blob(audio_blob) 

    except FileNotFoundError:
        # FFmpeg not found on PATH
        results = {
            'transcription': "Error: FFmpeg not found on PATH. Audio decoding failed.", 
            'emotion': 'Alert'
        }
    except Exception as e:
        # Catch any other decoding/processing failure
        print(f"Audio analysis failed: {e}")
        results = {
            'transcription': 'Audio input error. Please check your microphone.', 
            'emotion': 'Neutral'
        }

    # 2. Emit the results back to the client (to populate the input box)
    emit('audio_response', {
        'transcription': results['transcription'],
        'emotion': results['emotion']
    })


# Main 
if __name__ == '__main__':
    create_db()
    # Use socketio.run for Flask-SocketIO apps
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)