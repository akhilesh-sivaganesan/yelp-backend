# imports
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
# import fasttext
from collections import Counter
import os
import joblib
from sklearn.preprocessing import StandardScaler, Normalizer, MultiLabelBinarizer
from vectorizer import Vectorizer

def classify_by_closest_cluster(business, cluster_averages, classification_features, vectorizer, scaler=None):
    """
    Classify a given business by determining the closest cluster based on average feature vectors.

    Args:
        business (pd.Series): The data of the business to classify.

    Returns:
        int: The cluster label of the closest cluster.
    """
    # Construct the classification feature vector for the given business
    classification_vector = np.array([vectorizer.vectorize_business(business, classification_features)])
    if scaler:
        classification_vector = scaler.transform(classification_vector)

    # Find the closest cluster by computing distances to cluster averages
    closest_cluster = None
    min_distance = float('inf')
    for cluster_label, cluster_average in cluster_averages.items():
        distance = np.linalg.norm(classification_vector - cluster_average)
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster_label

    return closest_cluster

def clustering_insights(new_business, classification_feature):
    
    def load(filename):
        """Load a clustering pipeline"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    stored_classifier = load(f'./clustering/next_classifiers/stars(0.017-500)-{classification_feature}.pkl')
    stored_vectorizer = load('./clustering/updated_vectorizer2.pkl')
    classification_scaler = joblib.load(f'./clustering/scaler_{classification_feature}.joblib')
    
    classification = classify_by_closest_cluster(new_business, stored_classifier['avg_features'], [classification_feature], stored_vectorizer, classification_scaler)
    insights = stored_classifier['insights'][stored_classifier['insights']['cluster'] == classification]
    return insights.to_dict()



# new_business = {
#         'text': "good",
#         'categories': "Cafes, Coffee, Breakfast",
#         'latitude': 37.02,
#         'longitude': -88.8104,
#         'city': "San Francisco",
#         'hours': {
#             "Monday": "07:00-18:00",
#             "Tuesday": "07:00-18:00",
#             "Wednesday": "07:00-18:00",
#             "Thursday": "07:00-18:00",
#             "Friday": "07:00-18:00",
#             "Saturday": "08:00-16:00",
#             "Sunday": "08:00-16:00"
#         },
#         "stars": None,
#     }


# print(clustering_insights(new_business, 'sentiment'))

# myvectorizer = Vectorizer()
# # myvectorizer.top_categories = [
# #     'Restaurants',
# #     'Food',
# #     'Shopping',
# #     'Home Services',
# #     'Beauty & Spas',
# #     'Nightlife',
# #     'Health & Medical',
# #     'Local Services',
# #     'Bars',
# #     'Automotive'
# # ]
# pickle.dump(myvectorizer, open('./clustering/updated_vectorizer2.pkl', 'wb'))

