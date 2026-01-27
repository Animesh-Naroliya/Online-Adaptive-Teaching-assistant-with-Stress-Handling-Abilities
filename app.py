import time 
import os
from flask import Flask, jsonify, request, session, redirect, url_for, render_template, g 
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from database import db, User, Conversation, Message, Badge
from werkzeug.security import check_password_hash
from datetime import datetime

# --- EXTERNAL MODULE IMPORTS ---
from groqChatbot import llm_chatbot 
# REMOVED: from VoiceAnalysis.speechAnalyzer import analyze_audio_blob 
from video_analysis.video_analysis import analyze_video_frame
# ---

# Set this environment variable for local testing with HTTP
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a-default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

def create_db():
    with app.app_context():
        db.create_all()
        # Check if badges exist; if not, seed them
        if not Badge.query.first():
            print("Seeding Gem Gallery with Difficulty Badges...")
            badges = [
                # --- Difficulty Badges ---
                Badge(name="Novice Walker", description="Mastered the 'Very Easy' path.", icon_name="feather", rarity="Common", cost=10),
                Badge(name="Easy Going", description="Conquered 'Easy' quizzes.", icon_name="wind", rarity="Common", cost=25),
                Badge(name="Rising Star", description="Stepped up to 'Easy–Medium'.", icon_name="sunrise", rarity="Uncommon", cost=50),
                Badge(name="Solid Ground", description="Established 'Medium' mastery.", icon_name="anchor", rarity="Uncommon", cost=75),
                Badge(name="Mountain Climber", description="Scaled 'Medium–Hard'.", icon_name="trending-up", rarity="Rare", cost=150),
                Badge(name="Hard Rock", description="Crushed 'Hard' difficulty.", icon_name="shield", rarity="Rare", cost=250),
                Badge(name="Titanium Mind", description="Survived 'Very Hard'.", icon_name="cpu", rarity="Epic", cost=500),
                Badge(name="Grandmaster", description="Achieved 'Expert' status.", icon_name="award", rarity="Legendary", cost=1000),
                
                # --- Special Badges ---
                Badge(name="Gem Hoarder", description="Saved 500 Opal Gems.", icon_name="hexagon", rarity="Epic", cost=500),
                Badge(name="Streak Master", description="Kept a 7-day streak.", icon_name="zap", rarity="Rare", cost=200),
                Badge(name="Python Snake", description="A special badge for coders.", icon_name="code", rarity="Rare", cost=100)
            ]
            db.session.add_all(badges)
            db.session.commit()
        print("Database tables created!")

# --- UTILS ---
def get_current_user():
    uid = session.get('user_id')
    return User.query.get(uid) if uid else None

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session: return jsonify({'success': False, 'message': 'Auth required'}), 401
        if get_current_user() is None:
            session.pop('user_id', None)
            return jsonify({'success': False, 'message': 'Session invalid'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.before_request
def load_context():
    g.user = get_current_user()
    g.theme = g.user.theme if g.user else 'dark'

# --- ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/gamification')
@login_required
def gamification(): return render_template('gamification.html')

@app.route('/check_session')
def check_session():
    return jsonify({'is_authenticated': bool(g.user), 'username': g.user.username if g.user else None, 'theme': g.theme})

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not all([data.get('name'), data.get('username'), data.get('email'), data.get('password')]):
        return jsonify({'success': False, 'message': 'Missing fields'}), 400
    
    if User.query.filter((User.email == data['email']) | (User.username == data['username'])).first():
        return jsonify({'success': False, 'message': 'User exists'}), 409

    user = User(name=data['name'], username=data['username'], email=data['email'], password=data['password'])
    user.xp = 0
    user.level = 1
    user.opal_gems = 0
    user.streak = 1
    db.session.add(user)
    db.session.commit()
    session['user_id'] = user.id
    return jsonify({'success': True, 'message': 'User created'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter((User.email == data.get('identifier')) | (User.username == data.get('identifier'))).first()
    if user and user.check_password(data.get('password')):
        session['user_id'] = user.id
        return jsonify({'success': True, 'message': 'Login successful'}), 200
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

@app.route('/api/profile', methods=['GET', 'PUT'])
@login_required
def profile():
    if request.method == 'GET':
        u = g.user
        return jsonify({'success': True, 'name': u.name, 'username': u.username, 'email': u.email, 
                        'likes': u.likes.split(',') if u.likes else [], 
                        'dislikes': u.dislikes.split(',') if u.dislikes else [], 
                        'context': u.context})
    elif request.method == 'PUT':
        d = request.get_json()
        if d.get('likes') is not None: g.user.likes = ','.join(d['likes']) if isinstance(d['likes'], list) else d['likes']
        if d.get('dislikes') is not None: g.user.dislikes = ','.join(d['dislikes']) if isinstance(d['dislikes'], list) else d['dislikes']
        if d.get('context') is not None: g.user.context = d['context']
        db.session.commit()
        return jsonify({'success': True, 'message': 'Updated'})

# --- GAMIFICATION ---
@app.route('/api/gamification/stats', methods=['GET'])
@login_required
def get_stats():
    return jsonify({'success': True, 'level': g.user.level, 'current_xp': g.user.xp, 
                    'xp_needed': g.user.level * 100, 'opal_gems': g.user.opal_gems, 'streak': g.user.streak})

@app.route('/api/gamification/earn', methods=['POST'])
@login_required
def earn():
    d = request.get_json()
    g.user.xp += d.get('xp', 0)
    g.user.opal_gems += d.get('gems', 0)
    
    leveled = False
    while g.user.xp >= (g.user.level * 100):
        g.user.xp -= (g.user.level * 100)
        g.user.level += 1
        g.user.opal_gems += 20
        leveled = True
    
    db.session.commit()
    return jsonify({'success': True, 'new_level': g.user.level, 'leveled_up': leveled})

@app.route('/api/gamification/gallery', methods=['GET'])
@login_required
def gallery():
    badges = Badge.query.all()
    user_ids = [b.id for b in g.user.badges]
    data = [{'id': b.id, 'name': b.name, 'description': b.description, 'icon': b.icon_name, 
             'rarity': b.rarity, 'cost': b.cost, 'owned': b.id in user_ids, 
             'can_afford': g.user.opal_gems >= b.cost} for b in badges]
    return jsonify({'success': True, 'gallery': data})

@app.route('/api/gamification/buy/<int:bid>', methods=['POST'])
@login_required
def buy(bid):
    badge = Badge.query.get(bid)
    if not badge or badge in g.user.badges: return jsonify({'success': False, 'message': 'Invalid'}), 400
    if g.user.opal_gems < badge.cost: return jsonify({'success': False, 'message': 'Too expensive'}), 400
    
    g.user.opal_gems -= badge.cost
    g.user.badges.append(badge)
    db.session.commit()
    return jsonify({'success': True, 'new_gems': g.user.opal_gems, 'message': f'Unlocked {badge.name}!'})

@app.route('/api/gamification/generate_quiz', methods=['POST'])
@login_required
def generate_quiz_route():
    try:
        data = request.get_json() or {} 
        difficulty = data.get('difficulty', 'Medium') 
        
        last_conv = Conversation.query.filter_by(user_id=g.user.id).order_by(Conversation.created_at.desc()).first()
        
        chat_text = ""
        if last_conv:
            msgs = Message.query.filter_by(conversation_id=last_conv.id).order_by(Message.timestamp.desc()).limit(15).all()
            msgs.reverse()
            chat_text = "\n".join([f"{m.sender}: {m.content}" for m in msgs])
        else:
            chat_text = "No recent conversation. Please generate a general knowledge quiz."

        print(f"Generating {difficulty} quiz for User ID {g.user.id}")

        quiz_data = llm_chatbot.generate_quiz(chat_text, difficulty)
        
        if quiz_data:
            return jsonify({'success': True, 'quiz': quiz_data})
        else:
            return jsonify({'success': False, 'message': 'AI failed to format quiz.'}), 500

    except Exception as e:
        print(f"Quiz Server Error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# --- CHAT ---
@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    d = request.get_json()
    msg, cid, emo = d.get('message'), d.get('conversation_id'), d.get('emotion_detected')
    
    db.session.add(Message(conversation_id=cid, sender='user', content=msg, emotion_detected=emo))
    db.session.commit()
    
    # Updated: Passed only 'emotion_detected' since voice is removed
    u_data = {'username': g.user.username, 'context': g.user.context, 'likes': g.user.likes, 'current_emotion': emo}
    try: ai_resp = llm_chatbot.get_response(cid, msg, u_data)
    except: ai_resp = "Connection error."
        
    m = Message(conversation_id=cid, sender='vta', content=ai_resp)
    db.session.add(m)
    db.session.commit()
    return jsonify({'success': True, 'vta_response': ai_resp, 'message_id': m.id})

@app.route('/api/sessions', methods=['GET'])
@login_required
def sessions():
    s = Conversation.query.filter_by(user_id=g.user.id).order_by(Conversation.created_at.desc()).limit(10).all()
    return jsonify({'success': True, 'sessions': [{'id': x.id, 'title': x.title, 'created_at': x.created_at.strftime("%Y-%m-%d")} for x in s]})

@app.route('/api/sessions/new', methods=['POST'])
@login_required
def new_sess():
    c = Conversation(user_id=g.user.id, title=f"Session {datetime.now().strftime('%b %d')}")
    db.session.add(c)
    db.session.commit()
    w = Message(conversation_id=c.id, sender='vta', content="Hi! What shall we learn?")
    db.session.add(w)
    db.session.commit()
    return jsonify({'success': True, 'conversation_id': c.id, 'title': c.title, 'welcome_message': w.content})

@app.route('/api/sessions/<int:sid>/messages', methods=['GET'])
@login_required
def sess_msgs(sid):
    c = Conversation.query.filter_by(id=sid, user_id=g.user.id).first()
    if not c: return jsonify({'success': False}), 404
    msgs = Message.query.filter_by(conversation_id=sid).order_by(Message.timestamp.asc()).all()
    return jsonify({'success': True, 'messages': [{'sender': m.sender, 'content': m.content} for m in msgs], 'title': c.title})

# --- SOCKETS ---
@socketio.on('video_stream')
def vid(d): emit('video_response', {'emotion': analyze_video_frame(d.get('frame')) if d.get('frame') else 'Neutral'})

# REMOVED: @socketio.on('audio_stream') handler

if __name__ == '__main__':
    create_db()
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)