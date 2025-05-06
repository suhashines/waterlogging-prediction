from flask import Blueprint, jsonify, request
from db import db, auth_db

# Create a Blueprint for authentication routes
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({
            'message': 'Email and password are required',
            'status': 'error'
        }), 400
    
    # Create user with the database module
    user = auth_db.signup(email, password)
    
    if not user:
        return jsonify({
            'message': 'User already exists',
            'status': 'error'
        }), 400
    
    return jsonify({
        'user': user,
        'message': 'User created successfully',
        'status': 'success'
    })

@auth_bp.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({
            'message': 'Email and password are required',
            'status': 'error'
        }), 400
    
    # Sign in user with the database module
    result = auth_db.signin(email, password)
    
    if not result:
        return jsonify({
            'message': 'Invalid credentials',
            'status': 'error'
        }), 401
    
    return jsonify({
        **result,
        'message': 'Signed in successfully',
        'status': 'success'
    })

def init_app(app):
    app.register_blueprint(auth_bp)