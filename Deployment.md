# Deployment Guide

This guide provides instructions for deploying the Waterlogging Prediction System in various environments.

## Local Deployment

### Prerequisites
- Python 3.8+
- pip package manager
- Git (optional)

### Steps

1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/waterlogging_prediction.git
cd waterlogging_prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python train.py --data-dir data --model-type rf
```

5. Run the Flask API:
```bash
cd api
python app.py
```

The API will be available at `http://localhost:5000/`.

## Docker Deployment

### Prerequisites
- Docker
- Docker Compose (optional)

### Using Dockerfile

1. Build the Docker image:
```bash
docker build -t waterlogging-prediction:latest .
```

2. Run the container:
```bash
docker run -p 5000:5000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models waterlogging-prediction:latest
```

This will map the container's port 5000 to the host's port 5000 and mount the local data and models directories to the container.

### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - FLASK_ENV=production
```

Run the application:
```bash
docker-compose up -d
```

## Production Deployment

For production environments, consider the following recommendations:

### Using Gunicorn

Gunicorn is a production-ready WSGI server for Python web applications.

1. Install Gunicorn (already included in requirements.txt):
```bash
pip install gunicorn
```

2. Run the application with Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 api.app:app
```

### Using Nginx as a Reverse Proxy

For improved performance and security, use Nginx as a reverse proxy in front of Gunicorn.

1. Install Nginx:
```bash
sudo apt-get install nginx  # Ubuntu/Debian
# or
sudo yum install nginx      # CentOS/RHEL
```

2. Create an Nginx configuration file:
```
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. Enable the configuration and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/your_config /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

### Using Supervisor to Manage the Process

Supervisor can keep your application running and automatically restart it if it crashes.

1. Install Supervisor:
```bash
sudo apt-get install supervisor  # Ubuntu/Debian
# or
sudo yum install supervisor      # CentOS/RHEL
```

2. Create a configuration file:
```
[program:waterlogging-prediction]
command=/path/to/venv/bin/gunicorn --bind 0.0.0.0:5000 api.app:app
directory=/path/to/waterlogging_prediction
user=your_user
autostart=true
autorestart=true
stderr_logfile=/var/log/waterlogging_prediction.err.log
stdout_logfile=/var/log/waterlogging_prediction.out.log
```

3. Enable the configuration:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start waterlogging-prediction
```

## Cloud Deployment

### AWS Elastic Beanstalk

1. Install the EB CLI:
```bash
pip install awsebcli
```

2. Initialize your EB project:
```bash
eb init -p python-3.8 waterlogging-prediction
```

3. Create an environment and deploy:
```bash
eb create waterlogging-prediction-env
```

4. For subsequent deployments:
```bash
eb deploy
```

### Google Cloud Run

1. Install the Google Cloud SDK.

2. Build and push the Docker image:
```bash
gcloud builds submit --tag gcr.io/your-project-id/waterlogging-prediction
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy waterlogging-prediction \
  --image gcr.io/your-project-id/waterlogging-prediction \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure App Service

1. Install the Azure CLI.

2. Create an App Service Plan:
```bash
az appservice plan create --name waterlogging-prediction-plan --resource-group your-resource-group --sku B1
```

3. Create a Web App:
```bash
az webapp create --name waterlogging-prediction --resource-group your-resource-group --plan waterlogging-prediction-plan --runtime "PYTHON|3.8"
```

4. Deploy the application:
```bash
az webapp up --name waterlogging-prediction --resource-group your-resource-group
```

## Monitoring and Maintenance

### Logging

The application uses Python's built-in logging module. Logs are written to:
- Console output
- `app.log` file
- `training.log` file (for model training)

### Backup

Regularly backup your model files:
- `models/waterlogging_model.joblib`
- `models/risk_config.joblib`

### Updating the Model

To update the model with new data:
1. Add new CSV files to the `data/` directory
2. Run the training script:
```bash
python train.py --data-dir data --model-type rf
```

Alternatively, use the `/model/feedback` API endpoint to continuously improve the model with new observations.

### Security Considerations

1. Use HTTPS in production with a valid SSL certificate
2. Implement proper authentication for the API
3. Consider rate limiting to prevent abuse
4. Regularly update dependencies to address security vulnerabilities