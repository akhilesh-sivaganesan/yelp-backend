from flask import Flask, request, jsonify
from flask_cors import CORS
from cluster import clustering_insights
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def home():
    return 'Hello, World!'

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json  # Retrieve the JSON data sent by the form
    
    # process input to work with our existing model
    # --- Example Usage: Define a New Business ---
    new_business = {
        'text': "good",
        'categories': "Cafes, Coffee, Breakfast",
        'latitude': 37.02,
        'longitude': -88.8104,
        'city': "San Francisco",
        'hours': {
            "Monday": "07:00-18:00",
            "Tuesday": "07:00-18:00",
            "Wednesday": "07:00-18:00",
            "Thursday": "07:00-18:00",
            "Friday": "07:00-18:00",
            "Saturday": "08:00-16:00",
            "Sunday": "08:00-16:00"
        },
        "stars": None,
    }
    
    res = clustering_insights(new_business, 'sentiment')
    
    return str('heloo')
    # Log or process the data (for now, just return it back)
    # return jsonify(data)

if __name__ == '__main__':
    app.run()
    




