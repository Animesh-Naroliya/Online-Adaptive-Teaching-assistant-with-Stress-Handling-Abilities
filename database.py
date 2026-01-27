from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Initialize SQLAlchemy outside of the app instance
db = SQLAlchemy()

# Association table for User <-> Badge (Many-to-Many)
user_badges = db.Table('user_badges',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('badge_id', db.Integer, db.ForeignKey('badge.id'), primary_key=True),
    db.Column('earned_at', db.DateTime, default=datetime.utcnow)
)

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=True) 
    
    # New fields for social login
    social_id = db.Column(db.String(255), unique=True, nullable=True)
    
    # Profile Context
    likes = db.Column(db.Text, nullable=True)
    dislikes = db.Column(db.Text, nullable=True)
    context = db.Column(db.String(50), nullable=True)
    theme = db.Column(db.String(10), default='dark', nullable=False)

    # --- GAMIFICATION STATS ---
    xp = db.Column(db.Integer, default=0)
    level = db.Column(db.Integer, default=1)
    opal_gems = db.Column(db.Integer, default=0) # Currency
    streak = db.Column(db.Integer, default=0)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    conversations = db.relationship('Conversation', backref='user', lazy='dynamic')
    badges = db.relationship('Badge', secondary=user_badges, lazy='subquery',
        backref=db.backref('users', lazy=True))

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
        return check_password_hash(self.password_hash, password)

# Define Badge/Reward Model for the Gallery
class Badge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    icon_name = db.Column(db.String(50), nullable=False) # e.g., 'shield', 'star' (Feather icons)
    rarity = db.Column(db.String(20), default='Common') # Common, Rare, Legendary
    cost = db.Column(db.Integer, default=0) # Cost in Opal Gems

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