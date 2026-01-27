from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Initialize SQLAlchemy outside of the app instance
db = SQLAlchemy()

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=True) 
    
    # New fields for social login
    social_id = db.Column(db.String(255), unique=True, nullable=True)
    
    # Add new columns for user preferences and context
    likes = db.Column(db.Text, nullable=True)
    dislikes = db.Column(db.Text, nullable=True)
    context = db.Column(db.String(50), nullable=True)

    # New field for user settings (e.g., 'dark' or 'light')
    theme = db.Column(db.String(10), default='dark', nullable=False)

    # Relationship to Conversations
    conversations = db.relationship('Conversation', backref='user', lazy='dynamic')


    def __init__(self, name, username, email, password=None, social_id=None):
        self.name = name
        self.username = username
        self.email = email
        self.social_id = social_id
        if password:
            self.password_hash = generate_password_hash(password)

    def __repr__(self):
        return f'<User {self.username}>'

    def check_password(self, password):
        # NOTE: The existing app.py uses check_password_hash directly on user.password, which is a bug if not fixed in app.py. 
        # I'm fixing this by ensuring we check against the hash.
        return check_password_hash(self.password_hash, password)

# Define the Conversation/Session model
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to Messages
    
    messages = db.relationship('Message', backref='conversation', lazy='dynamic', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Conversation {self.title}>'

# Define the Message model for chat history
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    sender = db.Column(db.String(10), nullable=False)  # 'user' or 'vta'
    content = db.Column(db.Text, nullable=False)
    emotion_detected = db.Column(db.String(50), nullable=True) # Emotion recorded at the time of message
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Message {self.sender}: {self.content[:30]}>'