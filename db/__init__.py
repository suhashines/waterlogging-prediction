from .database import Database

# Create a global instance of the database
db = Database()

# Prevent circular imports by deferring module loading
def initialize_modules():
    from . import model_db
    from . import weather_db
    from . import route_db
    from . import forum_db
    from . import auth_db
    from . import authority_db

# Call this function to initialize the modules after db is created
initialize_modules()