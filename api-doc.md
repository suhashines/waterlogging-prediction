## API Documentation

### 1. Model Prediction Endpoints

#### 1.1 Predict Waterlogging Depth and Risk

**Endpoint**: `POST /model/predict/`

**Request Body**:
```json
{
    "station_code": "1",
    "rainfall": 10.5,
    "timestamp": "2023-05-01T12:30:00",
    "weather": 2,  
    "windspeed": 5.2  
}
```

**Response**:
```json
{
    "prediction": {
        "waterlogging_depth": 0.25,
        "risk_factor": {
            "risk_score": 0.45,
            "risk_level": "moderate",
            "amplification_factor": 0.23,
            "factors": {
                "elevation": 0.3,
                "impervious_cover": 0.8,
                "drainage": 0.4,
                "slope": 0.2,
                "proximity_to_water": 0.5
            }
        }
    },
    "message": "Prediction successful",
    "status": "success"
}
```

#### 1.2 Provide Feedback

**Endpoint**: `POST /model/feedback`

**Request Body**:
```json
{
    "station_code": "1",
    "rainfall": 10.5,
    "timestamp": "2023-05-01T12:30:00",
    "weather": 2,
    "windspeed": 5.2,
    "actual_waterdepth": 0.27
}
```

**Response**:
```json
{
    "message": "Feedback processed successfully",
    "status": "success",
    "details": {
        "previous_prediction": 0.25,
        "actual_value": 0.27,
        "error": 0.02
    }
}
```

#### 1.3 Get or Update Risk Weights

**Endpoint**: `GET /model/weights`

**Response**:
```json
{
    "weights": {
        "amplification_factor": 0.4,
        "elevation": 0.2,
        "impervious_cover": 0.1,
        "drainage": 0.15,
        "slope": 0.1,
        "proximity_to_water": 0.05
    },
    "message": "Current weights retrieved successfully",
    "status": "success"
}
```

**Endpoint**: `POST /model/weights`

**Request Body**:
```json
{
    "weights": {
        "amplification_factor": 0.5,
        "elevation": 0.2,
        "impervious_cover": 0.1,
        "drainage": 0.1,
        "slope": 0.05,
        "proximity_to_water": 0.05
    }
}
```

**Response**: Same as GET response but with updated weights.

#### 1.4 Get or Update Station Data

**Endpoint**: `GET /model/station-data?station_id=1`

**Response**:
```json
{
    "station_data": {
        "amplification_factor": 0.23,
        "elevation": 25.5,
        "impervious_cover": 0.85,
        "drainage_area": 200,
        "drainage_volume": 10000,
        "slope": 0.05,
        "proximity_to_water": 300
    },
    "message": "Station data retrieved successfully",
    "status": "success"
}
```

**Endpoint**: `POST /model/station-data`

**Request Body**:
```json
{
    "station_id": "1",
    "data": {
        "elevation": 25.5,
        "impervious_cover": 0.85,
        "drainage_area": 200,
        "drainage_volume": 10000,
        "slope": 0.05,
        "proximity_to_water": 300
    }
}
```

**Response**: Same as GET response but with updated station data.

### 2. Weather Data Endpoints

#### 2.1 Get Current Weather Data

**Endpoint**: `GET /weather/current?lat=23.8103&lon=90.4125`

**Response**:
```json
{
    "location": "Dhaka, Bangladesh",
    "currentCondition": "Heavy Rain",
    "temperature": 28,
    "rainfall": {
        "intensity": "heavy",
        "rate": 15.2
    },
    "timestamp": "2023-05-01T12:30:00Z",
    "message": "Weather data retrieved successfully",
    "status": "success"
}
```

#### 2.2 Get Weather Forecast

**Endpoint**: `GET /weather/forecast?lat=23.8103&lon=90.4125`

**Response**:
```json
{
    "location": "Dhaka, Bangladesh",
    "forecast": [
        {
            "time": "1h",
            "condition": "Heavy Rain",
            "temperature": 28,
            "rainfall": {
                "intensity": "heavy",
                "rate": 18.5
            }
        },
        {
            "time": "2h",
            "condition": "Moderate Rain",
            "temperature": 27,
            "rainfall": {
                "intensity": "medium",
                "rate": 8.2
            }
        },
        {
            "time": "3h",
            "condition": "Light Rain",
            "temperature": 26,
            "rainfall": {
                "intensity": "light",
                "rate": 3.1
            }
        }
    ],
    "message": "Weather forecast retrieved successfully",
    "status": "success"
}
```

#### 2.3 Get Weather Alerts

**Endpoint**: `GET /weather/alerts?lat=23.8103&lon=90.4125`

**Response**:
```json
{
    "location": "Dhaka, Bangladesh",
    "alerts": [
        {
            "type": "Flash waterlogging",
            "severity": "danger",
            "title": "Flash waterlogging Warning",
            "description": "Flash waterlogging is occurring or imminent in the warned area. Move to higher ground immediately.",
            "issued": "2023-05-01T10:30:00Z",
            "expires": "2023-05-01T16:30:00Z"
        }
    ],
    "message": "Weather alerts retrieved successfully",
    "status": "success"
}
```

### 3. Route Planning Endpoints

#### 3.1 Get Safe Routes

**Endpoint**: `POST /routes/plan`

**Request Body**:
```json
{
    "startLocation": "Dhaka",
    "endLocation": "Khulna",
    "timestamp": "2023-05-01T12:30:00"
}
```

**Response**:
```json
{
    "routes": [
        {
            "id": "route-1",
            "name": "Safest Route",
            "startLocation": {
                "lat": 23.8103,
                "lon": 90.4125,
                "name": "Dhaka"
            },
            "endLocation": {
                "lat": 22.8456,
                "lon": 89.5403,
                "name": "Khulna"
            },
            "segments": [
                {
                    "startPoint": {
                        "lat": 23.8103,
                        "lon": 90.4125
                    },
                    "endPoint": {
                        "lat": 23.3280,
                        "lon": 90.1764
                    },
                    "distance": 62000,
                    "duration": 4464,
                    "floodRisk": "none",
                    "roadType": "highway"
                },
                {
                    "startPoint": {
                        "lat": 23.3280,
                        "lon": 90.1764
                    },
                    "endPoint": {
                        "lat": 22.8456,
                        "lon": 89.5403
                    },
                    "distance": 248000,
                    "duration": 17856,
                    "floodRisk": "low",
                    "roadType": "major"
                }
            ],
            "totalDistance": 310000,
            "totalDuration": 22320,
            "safetyScore": 92,
            "safetyIssues": []
        },
        {
            "id": "route-2",
            "name": "Balanced Route",
            "startLocation": {
                "lat": 23.8103,
                "lon": 90.4125,
                "name": "Dhaka"
            },
            "endLocation": {
                "lat": 22.8456,
                "lon": 89.5403,
                "name": "Khulna"
            },
            "segments": [
                {
                    "startPoint": {
                        "lat": 23.8103,
                        "lon": 90.4125
                    },
                    "endPoint": {
                        "lat": 23.1640,
                        "lon": 89.9883
                    },
                    "distance": 90000,
                    "duration": 6480,
                    "floodRisk": "low",
                    "roadType": "highway"
                },
                {
                    "startPoint": {
                        "lat": 23.1640,
                        "lon": 89.9883
                    },
                    "endPoint": {
                        "lat": 22.8456,
                        "lon": 89.5403
                    },
                    "distance": 180000,
                    "duration": 12960,
                    "floodRisk": "medium",
                    "roadType": "major"
                }
            ],
            "totalDistance": 270000,
            "totalDuration": 19440,
            "safetyScore": 75,
            "safetyIssues": [
                {
                    "type": "waterlogging",
                    "description": "Moderate waterlogging reported near Faridpur",
                    "severity": "warning",
                    "location": {
                        "lat": 23.6065,
                        "lon": 89.8447
                    }
                }
            ]
        },
        {
            "id": "route-3",
            "name": "Shortest Route",
            "startLocation": {
                "lat": 23.8103,
                "lon": 90.4125,
                "name": "Dhaka"
            },
            "endLocation": {
                "lat": 22.8456,
                "lon": 89.5403,
                "name": "Khulna"
            },
            "segments": [
                {
                    "startPoint": {
                        "lat": 23.8103,
                        "lon": 90.4125
                    },
                    "endPoint": {
                        "lat": 23.4657,
                        "lon": 89.9764
                    },
                    "distance": 60000,
                    "duration": 4320,
                    "floodRisk": "medium",
                    "roadType": "highway"
                },
                {
                    "startPoint": {
                        "lat": 23.4657,
                        "lon": 89.9764
                    },
                    "endPoint": {
                        "lat": 23.0000,
                        "lon": 89.7532
                    },
                    "distance": 70000,
                    "duration": 5040,
                    "floodRisk": "high",
                    "roadType": "local"
                },
                {
                    "startPoint": {
                        "lat": 23.0000,
                        "lon": 89.7532
                    },
                    "endPoint": {
                        "lat": 22.8456,
                        "lon": 89.5403
                    },
                    "distance": 113000,
                    "duration": 8136,
                    "floodRisk": "extreme",
                    "roadType": "local"
                }
            ],
            "totalDistance": 243000,
            "totalDuration": 17496,
            "safetyScore": 45,
            "safetyIssues": [
                {
                    "type": "waterlogging",
                    "description": "Severe waterlogging reported on Jessore Highway",
                    "severity": "danger",
                    "location": {
                        "lat": 23.4657,
                        "lon": 89.9764
                    }
                },
                {
                    "type": "closure",
                    "description": "Road closure near Magura due to water levels",
                    "severity": "danger",
                    "location": {
                        "lat": 23.0000,
                        "lon": 89.7532
                    }
                }
            ]
        }
    ],
    "message": "Routes retrieved successfully",
    "status": "success"
}
```

### 4. Forum Endpoints

#### 4.1 Get All Posts

**Endpoint**: `GET /forum/posts?location=Dhaka&limit=10&offset=0`

**Response**:
```json
{
    "posts": [
        {
            "id": "post-1",
            "title": "waterlogging in Mirpur Area",
            "content": "Heavy waterlogging reported in parts of Mirpur. The main road is under 0.5m of water.",
            "location": "Mirpur, Dhaka",
            "created_at": "2023-05-01T10:30:00Z",
            "updated_at": "2023-05-01T10:30:00Z",
            "user_id": "user-1",
            "upvotes": 12,
            "downvotes": 2,
            "images": [
                {
                    "id": "img-1",
                    "image_url": "https://example.com/images/flood1.jpg"
                }
            ],
            "user_vote": null
        }
    ],
    "total": 1,
    "message": "Posts retrieved successfully",
    "status": "success"
}
```

#### 4.2 Create Post

**Endpoint**: `POST /forum/posts`

**Request Body**:
```json
{
    "title": "waterlogging in Mirpur Area",
    "content": "Heavy waterlogging reported in parts of Mirpur. The main road is under 0.5m of water.",
    "location": "Mirpur, Dhaka",
    "user_id": "user-1"
}
```

**Response**:
```json
{
    "post": {
        "id": "post-1",
        "title": "waterlogging in Mirpur Area",
        "content": "Heavy waterlogging reported in parts of Mirpur. The main road is under 0.5m of water.",
        "location": "Mirpur, Dhaka",
        "created_at": "2023-05-01T10:30:00Z",
        "updated_at": "2023-05-01T10:30:00Z",
        "user_id": "user-1",
        "upvotes": 0,
        "downvotes": 0
    },
    "message": "Post created successfully",
    "status": "success"
}
```

#### 4.3 Upload Post Image

**Endpoint**: `POST /forum/posts/:post_id/images`

**Request Body**: Multipart form data with image file

**Response**:
```json
{
    "image": {
        "id": "img-1",
        "post_id": "post-1",
        "image_url": "https://example.com/images/flood1.jpg",
        "created_at": "2023-05-01T10:35:00Z"
    },
    "message": "Image uploaded successfully",
    "status": "success"
}
```

#### 4.4 Vote on Post

**Endpoint**: `POST /forum/posts/:post_id/vote`

**Request Body**:
```json
{
    "user_id": "user-2",
    "vote_type": "upvote"
}
```

**Response**:
```json
{
    "post": {
        "id": "post-1",
        "upvotes": 13,
        "downvotes": 2,
        "user_vote": "upvote"
    },
    "message": "Vote recorded successfully",
    "status": "success"
}
```

### 5. Authentication Endpoints

#### 5.1 Sign Up

**Endpoint**: `POST /auth/signup`

**Request Body**:
```json
{
    "email": "user@example.com",
    "password": "securepassword"
}
```

**Response**:
```json
{
    "user": {
        "id": "user-3",
        "email": "user@example.com",
        "created_at": "2023-05-01T12:00:00Z"
    },
    "message": "User created successfully",
    "status": "success"
}
```

#### 5.2 Sign In

**Endpoint**: `POST /auth/signin`

**Request Body**:
```json
{
    "email": "user@example.com",
    "password": "securepassword"
}
```

**Response**:
```json
{
    "user": {
        "id": "user-3",
        "email": "user@example.com"
    },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "message": "Signed in successfully",
    "status": "success"
}
```

### 6. Authority Dashboard Endpoints

#### 6.1 Get All Stations

**Endpoint**: `GET /authority/stations`

**Response**:
```json
{
    "stations": [
        {
            "id": 1,
            "name": "Station A",
            "lat": 23.8103,
            "lon": 90.4125,
            "details": {
                "elevation": "11.2m",
                "landCover": "Urban/Impervious",
                "drainage": "Low",
                "slope": "1.8%",
                "proximity": "400m to Dhanmondi Lake"
            }
        },
        {
            "id": 2,
            "name": "Station B",
            "lat": 23.7000,
            "lon": 90.3750,
            "details": {
                "elevation": "9.6m",
                "landCover": "Mixed Residential",
                "drainage": "Moderate",
                "slope": "2.5%",
                "proximity": "700m to Buriganga River"
            }
        }
    ],
    "message": "Stations retrieved successfully",
    "status": "success"
}
```

#### 6.2 Get Station Data

**Endpoint**: `GET /authority/stations/:station_id/data?hours=3`

**Response**:
```json
{
    "station": {
        "id": 1,
        "name": "Station A",
        "lat": 23.8103,
        "lon": 90.4125
    },
    "data": [
        {
            "timestamp": "06:00",
            "waterlogging": 1.3,
            "rainfall": 10.0,
            "riskfactor": 2.0
        },
        {
            "timestamp": "06:30",
            "waterlogging": 1.2,
            "rainfall": 9.5,
            "riskfactor": 1.8
        },
        {
            "timestamp": "07:00",
            "waterlogging": 1.1,
            "rainfall": 9.0,
            "riskfactor": 1.7
        }
    ],
    "feedback": [
        {
            "user": "Hasan",
            "comment": "Water levels rise quickly here after heavy rain."
        },
        {
            "user": "Farzana",
            "comment": "Drainage improvements are working recently."
        }
    ],
    "message": "Station data retrieved successfully",
    "status": "success"
}
```

#### 6.3 Update Station Location

**Endpoint**: `POST /authority/stations/:station_id/location`

**Request Body**:
```json
{
    "lat": 23.8150,
    "lon": 90.4175
}
```

**Response**:
```json
{
    "station": {
        "id": 1,
        "name": "Station A",
        "lat": 23.8150,
        "lon": 90.4175
    },
    "message": "Station location updated successfully",
    "status": "success"
}
```