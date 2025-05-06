# Authentication data module
from . import db
import uuid
from datetime import datetime

# Define namespaces
USERS = 'auth_users'
TOKENS = 'auth_tokens'

# Initialize users
def init_users():
    return {
        'user@example.com': {
            'id': 'user-1',
            'email': 'user@example.com',
            'password': 'password123',
            'created_at': '2023-01-01T00:00:00Z'
        }
    }

# Initialize tokens
def init_tokens():
    return {}

# Register with the database
db.register(USERS, init_users)
db.register(TOKENS, init_tokens)

# Initialize data
db.initialize(USERS, init_users)
db.initialize(TOKENS, init_tokens)

# Authentication functions
def signup(email, password):
    """Register a new user"""
    users = db.get(USERS)
    
    if email in users:
        return None
    
    user_id = f"user-{len(users) + 1}"
    created_at = datetime.utcnow().isoformat() + 'Z'
    
    # Create new user
    users[email] = {
        'id': user_id,
        'email': email,
        'password': password,  # In a real app, this would be hashed
        'created_at': created_at
    }
    
    # Update database
    db.initialize(USERS, lambda: users)
    
    return {
        'id': user_id,
        'email': email,
        'created_at': created_at
    }

def signin(email, password):
    """Sign in an existing user"""
    users = db.get(USERS)
    tokens = db.get(TOKENS)
    
    user = users.get(email)
    if not user or user['password'] != password:
        return None
    
    # Generate token
    token = f"token-{uuid.uuid4()}"
    tokens[token] = user['id']
    
    # Update database
    db.initialize(TOKENS, lambda: tokens)
    
    return {
        'user': {
            'id': user['id'],
            'email': user['email']
        },
        'token': token
    }

def get_user_by_token(token):
    """Get a user by their authentication token"""
    tokens = db.get(TOKENS)
    users = db.get(USERS)
    
    user_id = tokens.get(token)
    if not user_id:
        return None
    
    # Find user by ID
    for email, user in users.items():
        if user['id'] == user_id:
            return {
                'id': user['id'],
                'email': user['email']
            }
    
    return None