from flask import Flask, request, jsonify
from flask_cors import CORS
from cluster import clustering_insights
from vectorizer import Vectorizer
from dt import PredictionManager

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
    return jsonify(res)
    # return str('heloo')
    # Log or process the data (for now, just return it back)
    # return jsonify(data)

@app.route('/api/dt', methods=['POST'])
def dt():
    data = request.json
    
    manager = PredictionManager(
        business_categories_file='./dt/business_df_headers.txt',
        shopping_categories_file='./dt/shopping_df_headers.txt',
        business_columns_file='./dt/business_columns.json',
        shopping_columns_file='./dt/shopping_columns.json',
        default_state='CA'
    )

    # Example JSON input (as a dictionary)
    input_json = {
        "businessCategories": [
            "ATV Rentals/Tours",
            "Vegan Restaurant"  # Example of multiple categories
        ],
        "businessState": "GA",
        "city": "Atlanta",
        "address": "848 Spring Street Northwest",
        "latitude": "33.778027300000005",
        "longitude": "-84.38922654305756",
        "attributes": {
            "AcceptsInsurance": True,
            "AgesAllowed": "19+",
            "Open24Hours": True,
            "BikeParking": True,
            "DietaryRestrictions": [
                "Gluten-Free",
                "Vegan"
            ],
            "HasTV": True,
            "WheelchairAccessible": True
        }
    }

    # Perform prediction
    
    predictions = manager.predict(input_json)
    print(predictions)
    return jsonify(predictions)


if __name__ == '__main__':
    app.run()
    




