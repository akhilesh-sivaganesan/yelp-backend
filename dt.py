import pandas as pd
import numpy as np
import joblib
import json
import os

# ----------------------------
# Prepare Features Functions
# ----------------------------

def prepare_features_business(df):
    """
    Prepare features for business (Restaurant) predictions by excluding specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing business data.
    
    Returns:
        pd.DataFrame: DataFrame with selected feature columns.
    """
    # Define columns to exclude
    exclude_cols = [
        'business_id', 'name', 'address', 'city', 'state', 'postal_code',
        'latitude', 'longitude', 'stars', 'is_open', 'attributes', 'hours',
        'review_count', 'average_reviews', 'star_success'  # Add any other columns that should be excluded
    ]
    
    # List of all feature columns
    all_features = df.columns.tolist()
    
    # Remove excluded columns
    feature_cols = [col for col in all_features if col not in exclude_cols]
    
    X = df[feature_cols]
    return X

def prepare_features_shopping(df):
    """
    Prepare features for shopping predictions by excluding specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing shopping data.
    
    Returns:
        pd.DataFrame: DataFrame with selected feature columns.
    """
    # Define columns to exclude
    exclude_cols = [
        'business_id', 'name', 'address', 'city', 'state', 'postal_code',
        'latitude', 'longitude', 'stars', 'is_open', 'attributes', 'hours',
        'review_count', 'average_reviews', 'star_success'  # Add any other columns that should be excluded
    ]
    
    # List of all feature columns
    all_features = df.columns.tolist()
    
    # Remove excluded columns
    feature_cols = [col for col in all_features if col not in exclude_cols]
    
    X = df[feature_cols]
    return X

# ----------------------------
# Utility Functions
# ----------------------------
def load_json_data(file_path):
    """
    Load JSON data from a given file path.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Loaded JSON data or an empty dictionary if the file doesn't exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: JSON file not found at {file_path}")
        return {}

def load_categories(file_path):
    """
    Load categories from a text file, one category per line.
    
    Args:
        file_path (str): Path to the category text file.
    
    Returns:
        set: A set of categories.
    """
    with open(file_path, 'r') as file:
        categories = set(line.strip() for line in file if line.strip())
    return categories

def load_columns_info(file_path):
    """
    Load column information from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing column info.
    
    Returns:
        dict: Dictionary with column names as keys and data types as values.
    """
    with open(file_path, 'r') as f:
        columns_info = json.load(f)
    return columns_info

def convert_numpy_types(obj):
    """
    Recursively convert NumPy data types to native Python data types.

    Args:
        obj: The object to convert.

    Returns:
        The converted object with native Python data types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ----------------------------
# Prediction Functions
# ----------------------------

def predict_from_json_business(json_obj, state, targets=['is_open', 'star_success'], columns_info=None):
    """
    Predict for business (Restaurant) category using the provided JSON object.
    
    Args:
        json_obj (dict): Input JSON object with business data.
        state (str): State code for model selection.
        targets (list): List of target variables to predict.
        columns_info (dict): Dictionary with column names and data types.
    
    Returns:
        dict: Predictions for each target from different models.
    """
    if columns_info is None:
        raise ValueError("columns_info must be provided.")
    
    predictions = {}
    
    # Initialize the business DataFrame with default values
    business_df = {}
    for column, col_type in columns_info.items():
        keys = column.split('.')
        value = json_obj
        for key in keys:
            value = value.get(key, None)
            if value is None:
                break
        if value is None:
            if col_type in ["int", "float"]:
                business_df[column] = 0
            elif col_type == "bool":
                business_df[column] = False
            else:
                business_df[column] = ""
        else:
            # Handle nested lists (e.g., DietaryRestrictions)
            if isinstance(value, list):
                business_df[column] = ','.join(value)
            elif col_type == "float":
                try:
                    business_df[column] = float(value)
                except ValueError:
                    business_df[column] = 0.0
            elif col_type == "int":
                try:
                    business_df[column] = int(value)
                except ValueError:
                    business_df[column] = 0
            else:
                business_df[column] = value
    
    business_df = pd.DataFrame([business_df])
    
    # Prepare features
    X = prepare_features_business(business_df)
    
    # Load models
    state_folder = f'./dt/models_and_plots/{state}'
    
    for target in targets:
        predictions[target] = {}
        model_path = f"{state_folder}/{target}_model.pkl"
        logistic_model_path = f"{state_folder}/{target}_logistic_model.pkl"  # Only for 'is_open'
        
        # Predict using the RandomForest model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            rf_prediction = model.predict(X)[0]
            predictions[target]['RandomForest'] = rf_prediction
        else:
            print(f"RandomForest model not found for state {state} at {model_path}")
        
        # Predict using the Logistic model for 'is_open' target
        if target == 'is_open':
            if os.path.exists(logistic_model_path):
                logistic_model = joblib.load(logistic_model_path)
                lr_prediction = logistic_model.predict(X)[0]
                predictions[target]['LogisticRegression'] = lr_prediction
            else:
                print(f"LogisticRegression model not found for state {state} at {logistic_model_path}")
    
    return predictions

def predict_from_json_shopping(json_obj, state, targets=['is_open', 'star_success'], columns_info=None):
    """
    Predict for shopping category using the provided JSON object.
    
    Args:
        json_obj (dict): Input JSON object with shopping data.
        state (str): State code for model selection.
        targets (list): List of target variables to predict.
        columns_info (dict): Dictionary with column names and data types.
    
    Returns:
        dict: Predictions for each target from different models.
    """
    if columns_info is None:
        raise ValueError("columns_info must be provided.")
    
    predictions = {}
    
    # Initialize the shopping DataFrame with default values
    shopping_df = {}
    for column, col_type in columns_info.items():
        keys = column.split('.')
        value = json_obj
        for key in keys:
            value = value.get(key, None)
            if value is None:
                break
        if value is None:
            if col_type in ["int", "float"]:
                shopping_df[column] = 0
            elif col_type == "bool":
                shopping_df[column] = False
            else:
                shopping_df[column] = ""
        else:
            # Handle nested lists (e.g., DietaryRestrictions)
            if isinstance(value, list):
                shopping_df[column] = ','.join(value)
            elif col_type == "float":
                try:
                    shopping_df[column] = float(value)
                except ValueError:
                    shopping_df[column] = 0.0
            elif col_type == "int":
                try:
                    shopping_df[column] = int(value)
                except ValueError:
                    shopping_df[column] = 0
            else:
                shopping_df[column] = value
    
    shopping_df = pd.DataFrame([shopping_df])
    
    # Prepare features
    X = prepare_features_shopping(shopping_df)
    
    # Load models
    state_folder = f'./dt/shop_models_and_plots/{state}'
    
    for target in targets:
        predictions[target] = {}
        model_path = f"{state_folder}/{target}_model.pkl"
        logistic_model_path = f"{state_folder}/{target}_logistic_model.pkl"  # Only for 'is_open'
        
        # Predict using the RandomForest model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            rf_prediction = model.predict(X)[0]
            predictions[target]['RandomForest'] = rf_prediction
        else:
            print(f"RandomForest model not found for state {state} at {model_path}")
        
        # Predict using the Logistic model for 'is_open' target
        if target == 'is_open':
            if os.path.exists(logistic_model_path):
                logistic_model = joblib.load(logistic_model_path)
                lr_prediction = logistic_model.predict(X)[0]
                predictions[target]['LogisticRegression'] = lr_prediction
            else:
                print(f"LogisticRegression model not found for state {state} at {logistic_model_path}")
    
    return predictions

# ----------------------------
# Prediction Manager Class
# ----------------------------

class PredictionManager:
    """
    Manages the prediction process by determining the appropriate predictor
    (business or shopping) based on category overlaps and handling predictions.
    """
    def __init__(self, business_categories_file, shopping_categories_file, 
                 business_columns_file, shopping_columns_file, default_state='CA'):
        self.business_categories = load_categories(business_categories_file)
        self.shopping_categories = load_categories(shopping_categories_file)
        self.default_state = default_state
        self.business_columns_info = load_columns_info(business_columns_file)
        self.shopping_columns_info = load_columns_info(shopping_columns_file)
    
    def determine_predictor(self, input_categories):
        """
        Determine whether to use business or shopping predictor based on category overlaps.
        
        Args:
            input_categories (list): List of categories from input JSON.
        
        Returns:
            str: 'business' or 'shopping'
        """
        business_overlap = len(set(input_categories) & self.business_categories)
        shopping_overlap = len(set(input_categories) & self.shopping_categories)

        if business_overlap > shopping_overlap:
            return 'business'
        elif shopping_overlap > business_overlap:
            return 'shopping'
        else:
            # Tiebreaker: default to 'business'
            return 'business'
    
    def filter_categories(self, json_obj, predictor):
        """
        Remove categories not relevant to the chosen predictor.
        
        Args:
            json_obj (dict): Input JSON object.
            predictor (str): 'business' or 'shopping'
        
        Returns:
            dict: Filtered JSON object with relevant categories.
        """
        if predictor == 'business':
            relevant_categories = self.business_categories
            # irrelevant_categories = self.shopping_categories  # Not used
        else:
            relevant_categories = self.shopping_categories
            # irrelevant_categories = self.business_categories  # Not used

        # Filter out irrelevant categories
        filtered_categories = list(set(json_obj.get("businessCategories", [])) & relevant_categories)
        json_obj["businessCategories"] = filtered_categories

        return json_obj
    
    def get_state(self, state, predictor):
        """
        Determine the appropriate state for model loading, defaulting to 'CA' if necessary.
        
        Args:
            state (str): Provided state code.
            predictor (str): 'business' or 'shopping'
        
        Returns:
            str: Valid state code with existing model directory.
        """
        if predictor == 'business':
            folder = f'./dt/models_and_plots/{state}'
        else:
            folder = f'./dt/shop_models_and_plots/{state}'

        if os.path.exists(folder):
            return state
        else:
            return self.default_state
    
    def predict(self, json_input):
        """
        Perform prediction based on input JSON and include feature importances and coefficients.
        
        Args:
            json_input (dict): Input JSON object with business data.
        
        Returns:
            dict: Combined predictions and corresponding feature data.
        """
        # Determine which predictor to use
        input_categories = json_input.get("businessCategories", [])
        predictor = self.determine_predictor(input_categories)

        # Filter categories based on the chosen predictor
        filtered_json = self.filter_categories(json_input.copy(), predictor)

        # Determine state, default to CA if not provided or invalid
        state = filtered_json.get("businessState", self.default_state)
        state = self.get_state(state, predictor)

        # Define targets to predict
        targets = ['is_open', 'star_success']

        # Make prediction using the appropriate predictor
        if predictor == 'business':
            predictions = predict_from_json_business(
                filtered_json, 
                state, 
                targets=targets,
                columns_info=self.business_columns_info
            )
            state_folder = f'./dt/models_and_plots/{state}'
        else:
            predictions = predict_from_json_shopping(
                filtered_json, 
                state, 
                targets=targets,
                columns_info=self.shopping_columns_info
            )
            state_folder = f'./dt/shop_models_and_plots/{state}'

        # Initialize dictionaries to hold feature data
        feature_data = {}
        
        for target in targets:
            feature_data[target] = {}
            
            # Load Feature Importances
            fi_path = os.path.join(state_folder, f"{target}_feature_importance.json")
            feature_importances = load_json_data(fi_path)
            if feature_importances:
                feature_data[target]['feature_importances'] = feature_importances
            
            # Load Logistic Coefficients (only for 'is_open')
            if target == 'is_open':
                coeff_path = os.path.join(state_folder, f"{target}_logistic_coefficients.json")
                logistic_coefficients = load_json_data(coeff_path)
                if logistic_coefficients:
                    feature_data[target]['logistic_coefficients'] = logistic_coefficients
        
        # Combine predictions with feature data
        combined_response = {
            'state': state,  # The state used for prediction (could be default 'CA')
            'predictions': predictions,
            'feature_data': feature_data
        }
        combined_response = convert_numpy_types(combined_response)

        return combined_response 
# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    # Initialize the PredictionManager with category and columns files
    manager = PredictionManager(
        business_categories_file='business_df_headers.txt',
        shopping_categories_file='shopping_df_headers.txt',
        business_columns_file='business_columns.json',
        shopping_columns_file='shopping_columns.json',
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
