from flask import Flask, render_template, request, jsonify
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from jinja2 import Environment, FileSystemLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import pickle
import datetime




app = Flask(__name__)
app.jinja_env.cache = {}

analyzer = SentimentIntensityAnalyzer()
# Load dataset once when server starts
df = pd.read_csv('netflix_titles_100.csv')

# Fill missing values and limit rows to show (optional)
df = df[['title', 'release_year', 'rating', 'description']].dropna().head(12)
tfidf = TfidfVectorizer()
tfidf_matrix=tfidf.fit_transform(df['description'])

def recommendation(index, top_n=5):
    tfidf_movie = tfidf_matrix[index]
    desc_cosine_sim = cosine_similarity(tfidf_matrix, tfidf_movie)
    recommend_indices = np.argsort(desc_cosine_sim.flatten())[::-1][1:top_n+1]
    return df.iloc[recommend_indices].to_dict(orient='records')

# Step 1: Fetch current weather
API_KEY = 'cea962528ea2296e50a65b42c009e86c'
CITY = 'kochi'
URL = f'http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric'

response = requests.get(URL)
data = response.json()

# Step 2: Extract values
temperature = data['main']['temp']
humidity = data['main']['humidity']
pressure = data['main']['pressure']  # in hPa, equivalent to millibars
wind_speed = data['wind']['speed'] * 3.6  # m/s to km/h
wind_bearing = data['wind'].get('deg', 0)  # default to 0 if missing

# Optional: Use defaults or estimate if not provided by API
apparent_temp = temperature  # assume equal if unavailable
visibility = data.get('visibility', 10000) / 1000  # in km
cloud_cover = data['clouds'].get('all', 0) / 100  # convert 0–100 to 0–1 scale

# Categorical fields (convert to strings)
summary = data['weather'][0]['main']  # e.g., 'Clear'
precip_type = 'rain' if 'rain' in data else 'none'
daily_summary = data['weather'][0]['description']

# Date features
today = datetime.datetime.now()
month = today.month
day = today.day
year = today.year
dayofweek = today.weekday()  # Monday=0

# Step 3: Create a DataFrame with a single row
import pandas as pd

input_df = pd.DataFrame([{
    'Summary': summary,
    'Precip Type': precip_type,
    'Temperature (C)': temperature,
    'Apparent Temperature (C)': apparent_temp,
    'Humidity': humidity,
    'Wind Speed (km/h)': wind_speed,
    'Wind Bearing (degrees)': wind_bearing,
    'Visibility (km)': visibility,
    'Loud Cover': cloud_cover,
    'Pressure (millibars)': pressure,
    'Daily Summary': daily_summary,
    'month': month,
    'day': day,
    'year': year,
    'dayofweek': dayofweek
}])

# Optional: apply same preprocessing (LabelEncoding, OneHot, etc.) if model needs it

# Step 4: Load model and predict
with open('models/weather_model.pkl', 'rb') as f:
    model = pickle.load(f)

# If your model expects encoded columns, you must preprocess `input_df` exactly like your training data
prediction = model.predict(input_df)

print("Model Prediction:", prediction)

def interpret_weather(temp_c):
    if temp_c >= 30:
        return (
            "Blazing sun and summer heat dominate the skies.",
            "It's a scorcher! Time for a summer blockbuster. Grab your shades – it's the kind of day where heroes save the world under a blazing sun."
        )
    elif 20 <= temp_c < 30:
        return (
            "Clear blue skies with a touch of cinematic charm.",
            "Clear skies ahead — perfect for a feel-good film. Like the calm before a rom-com twist, everything feels just right."
        )
    elif 10 <= temp_c < 20:
        return (
            "Moody clouds drifting like a plot waiting to unfold.",
            "Cloudy – just like a slow-burn mystery thriller! Feels like the setup for a plot twist no one saw coming."
        )
    elif 0 <= temp_c < 10:
        return (
            "Crisp and chilly air sets the stage for romance.",
            "Chilly vibes call for a cozy rom-com. Time to wrap up, sip cocoa, and watch two strangers fall in love by accident."
        )
    else:
        return (
            "Snow-kissed silence blankets everything like a drama's final scene.",
            "Snowy and serene — maybe a classic drama? The kind of day where emotions run deep and snow falls in slow motion."
        )


weather, quote = interpret_weather(prediction)

@app.route('/')
def index():
    movie_row = df.sample(1)
    movie = movie_row.to_dict(orient='records')[0]
    movie_index = movie_row.index[0]
    recommendations = recommendation(movie_index)
    return render_template('index.html', movie=movie,  weather=weather,
        quote=quote,recommendations=recommendations)




@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    review_text = request.form['review_text']
    score = analyzer.polarity_scores(review_text)
    compound = score['compound']
    sentiment = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    
    movie_row = df.sample(1)
    movie = movie_row.to_dict(orient='records')[0]
    movie_index = movie_row.index[0]
    recommendations = recommendation(movie_index)
    
    return render_template('index.html', movie=movie,weather=weather,
        quote=quote, sentiment=sentiment, recommendations=recommendations)



# Placeholder routes
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    return jsonify({"sentiment": "positive"})

@app.route('/recommend_movies', methods=['POST'])
def recommend_movies():
    return jsonify({"movies": ["Inception", "Interstellar", "The Matrix"]})

@app.route('/detect_anomaly', methods=['POST'])
def detect_anomaly():
    return jsonify({"anomaly": True})

# Add more endpoints as models are added

if __name__ == '__main__':
    app.run(debug=True)
