# Forum data module
from . import db
import random
from datetime import datetime, timedelta
import uuid

# Define namespaces
POSTS = 'posts'
POST_IMAGES = 'post_images'
USER_VOTES = 'user_votes'
USERS = 'users'

# Sample users
def init_users():
    return {
        'user-1': {"id": "user-1", "email": "hasan@example.com", "password": "password123"},
        'user-2': {"id": "user-2", "email": "farzana@example.com", "password": "password123"},
        'user-3': {"id": "user-3", "email": "tariq@example.com", "password": "password123"},
        'user-4': {"id": "user-4", "email": "mitu@example.com", "password": "password123"},
        'user-5': {"id": "user-5", "email": "rayhan@example.com", "password": "password123"}
    }

# Sample locations
SAMPLE_LOCATIONS = [
    "Mirpur, Dhaka",
    "Dhanmondi, Dhaka",
    "Gulshan, Dhaka",
    "Khulna City",
    "Chittagong Port Area",
    "Sylhet City"
]

# Sample post titles and content templates
SAMPLE_TITLES = [
    "waterlogging in {location}",
    "Water levels rising in {location}",
    "Road conditions in {location}",
    "Heavy rain impact in {location}",
    "Drainage issues in {location}"
]

SAMPLE_CONTENTS = [
    "There is significant waterlogging in {location}. Water level is approximately {depth}m deep. Be careful when traveling through this area.",
    "Water levels are rising quickly in {location} due to continuous rain. Main roads have {depth}m of water.",
    "Current road conditions in {location} are challenging. Several areas have {depth}m deep water pools. Alternative routes recommended.",
    "Heavy rain has impacted {location} significantly. Streets are flooded with {depth}m of water in some areas.",
    "Drainage systems in {location} are overwhelmed. Water accumulation of {depth}m reported in low-lying areas."
]

# Initialize dummy data
def init_posts():
    users = list(init_users().values())
    posts = []
    
    for i in range(10):
        post_id = f"post-{i+1}"
        user = random.choice(users)
        location = random.choice(SAMPLE_LOCATIONS)
        title_template = random.choice(SAMPLE_TITLES)
        content_template = random.choice(SAMPLE_CONTENTS)
        
        # Random water depth between 0.1 and 1.0 meters
        depth = round(random.uniform(0.1, 1.0), 1)
        
        # Format title and content
        title = title_template.format(location=location)
        content = content_template.format(location=location, depth=depth)
        
        # Random timestamp within the last 3 days
        created_at = (datetime.utcnow() - timedelta(days=random.uniform(0, 3))).isoformat() + 'Z'
        
        # Random votes
        upvotes = random.randint(0, 20)
        downvotes = random.randint(0, 5)
        
        post = {
            "id": post_id,
            "title": title,
            "content": content,
            "location": location,
            "created_at": created_at,
            "updated_at": created_at,
            "user_id": user["id"],
            "upvotes": upvotes,
            "downvotes": downvotes
        }
        
        posts.append(post)
    
    return posts

def init_post_images():
    images = {}
    posts = init_posts()
    
    for post in posts:
        if random.random() > 0.5:
            post_images = []
            num_images = random.randint(1, 3)
            
            for i in range(num_images):
                image_id = f"img-{uuid.uuid4()}"
                image_url = f"https://example.com/images/waterlogging{random.randint(1, 10)}.jpg"
                
                post_images.append({
                    "id": image_id,
                    "image_url": image_url
                })
                
            images[post['id']] = post_images
    
    return images

def init_user_votes():
    return {}

# Register with the database
db.register(POSTS, init_posts)
db.register(POST_IMAGES, init_post_images)
db.register(USER_VOTES, init_user_votes)
db.register(USERS, init_users)

# Initialize data
db.initialize(POSTS, init_posts)
db.initialize(POST_IMAGES, init_post_images)
db.initialize(USER_VOTES, init_user_votes)
db.initialize(USERS, init_users)

# Forum functions
def get_posts(location=None, limit=10, offset=0, user_id=None):
    """Get forum posts with optional filtering"""
    posts = db.get(POSTS)
    
    # Filter by location if provided
    if location:
        filtered_posts = [p for p in posts if location.lower() in p['location'].lower()]
    else:
        filtered_posts = posts
    
    # Apply pagination
    paginated_posts = filtered_posts[offset:offset+limit]
    
    # Add images to posts
    post_images = db.get(POST_IMAGES)
    user_votes = db.get(USER_VOTES)
    
    result_posts = []
    for post in paginated_posts:
        post_copy = post.copy()
        post_copy['images'] = post_images.get(post['id'], [])
        
        # Add user_vote if user_id is provided
        if user_id:
            vote_key = f"{user_id}:{post['id']}"
            post_copy['user_vote'] = user_votes.get(vote_key)
        else:
            post_copy['user_vote'] = None
            
        result_posts.append(post_copy)
    
    return {
        'posts': result_posts,
        'total': len(filtered_posts)
    }

def create_post(title, content, location, user_id):
    """Create a new forum post"""
    posts = db.get(POSTS)
    post_id = f"post-{len(posts) + 1}"
    created_at = datetime.utcnow().isoformat() + 'Z'
    
    new_post = {
        "id": post_id,
        "title": title,
        "content": content,
        "location": location,
        "created_at": created_at,
        "updated_at": created_at,
        "user_id": user_id,
        "upvotes": 0,
        "downvotes": 0
    }
    
    posts.append(new_post)
    db.initialize(POSTS, lambda: posts)
    
    return new_post

def upload_image(post_id):
    """Upload an image to a post"""
    # In a real implementation, we would handle file upload
    # For our mock API, we'll just generate a dummy image
    
    image_id = f"img-{uuid.uuid4()}"
    image_url = f"https://example.com/images/waterlogging{random.randint(1, 10)}.jpg"
    
    image = {
        "id": image_id,
        "post_id": post_id,
        "image_url": image_url,
        "created_at": datetime.utcnow().isoformat() + 'Z'
    }
    
    post_images = db.get(POST_IMAGES)
    if post_id not in post_images:
        post_images[post_id] = []
    post_images[post_id].append(image)
    
    db.initialize(POST_IMAGES, lambda: post_images)
    
    return image

def vote_on_post(post_id, user_id, vote_type):
    """Record a user vote on a post"""
    posts = db.get(POSTS)
    user_votes = db.get(USER_VOTES)
    
    # Find the post
    post = next((p for p in posts if p['id'] == post_id), None)
    if not post:
        return None
    
    # Check if user has already voted
    vote_key = f"{user_id}:{post_id}"
    existing_vote = user_votes.get(vote_key)
    
    if existing_vote == vote_type:
        # Remove the vote
        if vote_key in user_votes:
            del user_votes[vote_key]
        post[f"{vote_type}s"] = max(0, post[f"{vote_type}s"] - 1)
        user_vote = None
    elif existing_vote:
        # Change vote type
        user_votes[vote_key] = vote_type
        post[f"{existing_vote}s"] = max(0, post[f"{existing_vote}s"] - 1)
        post[f"{vote_type}s"] += 1
        user_vote = vote_type
    else:
        # New vote
        user_votes[vote_key] = vote_type
        post[f"{vote_type}s"] += 1
        user_vote = vote_type
    
    # Update the database
    db.initialize(POSTS, lambda: posts)
    db.initialize(USER_VOTES, lambda: user_votes)
    
    return {
        'id': post_id,
        'upvotes': post['upvotes'],
        'downvotes': post['downvotes'],
        'user_vote': user_vote
    }