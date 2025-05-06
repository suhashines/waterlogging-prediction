from flask import Blueprint, jsonify, request
from db import db, forum_db

# Create a Blueprint for forum routes
forum_bp = Blueprint('forum', __name__, url_prefix='/forum')

@forum_bp.route('/posts', methods=['GET'])
def get_posts():
    # Get query parameters
    location = request.args.get('location')
    limit = int(request.args.get('limit', 10))
    offset = int(request.args.get('offset', 0))
    user_id = request.args.get('user_id')
    
    # Get posts from the database module
    result = forum_db.get_posts(location, limit, offset, user_id)
    
    return jsonify({
        **result,
        'message': 'Posts retrieved successfully',
        'status': 'success'
    })

@forum_bp.route('/posts', methods=['POST'])
def create_post():
    data = request.get_json()
    
    # Extract data from request
    title = data.get('title', '')
    content = data.get('content', '')
    location = data.get('location', '')
    user_id = data.get('user_id', '')
    
    # Create post with the database module
    post = forum_db.create_post(title, content, location, user_id)
    
    return jsonify({
        'post': post,
        'message': 'Post created successfully',
        'status': 'success'
    })

@forum_bp.route('/posts/<post_id>/images', methods=['POST'])
def upload_image(post_id):
    # Upload image with the database module
    image = forum_db.upload_image(post_id)
    
    return jsonify({
        'image': image,
        'message': 'Image uploaded successfully',
        'status': 'success'
    })

@forum_bp.route('/posts/<post_id>/vote', methods=['POST'])
def vote_on_post(post_id):
    data = request.get_json()
    
    # Extract data from request
    user_id = data.get('user_id')
    vote_type = data.get('vote_type')
    
    if not user_id or not vote_type or vote_type not in ['upvote', 'downvote']:
        return jsonify({
            'message': 'Invalid request data',
            'status': 'error'
        }), 400
    
    # Vote on post with the database module
    result = forum_db.vote_on_post(post_id, user_id, vote_type)
    
    if not result:
        return jsonify({
            'message': 'Post not found',
            'status': 'error'
        }), 404
        
    return jsonify({
        'post': result,
        'message': 'Vote recorded successfully',
        'status': 'success'
    })

def init_app(app):
    app.register_blueprint(forum_bp)