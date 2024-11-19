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


class Vectorizer:
    def __init__(self):
        # Instance-level attributes (instead of class-level)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.top_categories = None
        self.kinds = ['location', 'hours', 'categories', 'stars', 'sentiment']

    def category_vector(self, business):
        business_categories = set(business['categories'].split(', ')) if isinstance(business['categories'], str) else set()
        vector = np.zeros(len(self.top_categories))
        for category in business_categories.intersection(set(self.top_categories)):
            idx = self.top_categories.index(category)
            vector[idx] = 1
        return vector
    
    def hours_vector(self, business):
        if not business['hours']:
            return np.zeros(2)
        open_weekends = int('Saturday' in business['hours'] or 'Sunday' in business['hours'])
        open_late = int(any(h and h.split('-')[1] >= '21:00' for h in business['hours'].values()))
        return np.array([open_weekends, open_late])
    
    def location_vector(self, business):
        return np.array([business['latitude'], business['longitude']])
    
    def stars_vector(self, business):
        return np.array([business['average_star_rating']])
    
    def sentiment_vector(self, text, max_length=512, batch_size=64):
        """
        Compute sentiment vector for a large text by processing it in chunks.

        Args:
            text (str): The input text blob.
            max_length (int): Maximum token length per chunk.
            batch_size (int): Number of chunks to process in a batch.

        Returns:
            np.ndarray: Aggregated sentiment probabilities [prob_negative, prob_positive].
        """
        if not text or not isinstance(text, str):
            return np.zeros(2)  # Return neutral sentiment for empty/missing text

        # Split text into chunks based on tokenizer's max length
        tokens = self.sentiment_tokenizer(text, return_tensors="pt", truncation=False, padding=False).input_ids[0]
        chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        
        all_probabilities = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_model.to(device)
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]

            # Prepare batch for model
            batch_inputs = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(chunk) for chunk in batch_chunks], 
                batch_first=True
            ).to(device)

            with torch.no_grad():
                # Process batch and compute logits
                outputs = self.sentiment_model(input_ids=batch_inputs)
                logits = outputs.logits

            # Convert logits to probabilities
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            all_probabilities.extend(probabilities)

        # Aggregate probabilities across all chunks (e.g., averaging)
        aggregated_probabilities = np.mean(all_probabilities, axis=0)
        return aggregated_probabilities
    
    def save_top_categories(self, business_data):
        # Split the categories for each business into a list
        categories_df = business_data['categories'].fillna('').apply(lambda x: x.split(', '))
        
        # Flatten the list of all categories and count occurrences
        category_counts = Counter(category for categories in categories_df for category in categories)
        
        # Get the 10 most common categories
        top_categories = np.array([(cat, count) for cat, count in category_counts.most_common(10)])
        for cat, count in top_categories:
            print(f"{cat}: {count}")
        self.top_categories = top_categories[:, 0].tolist()
        
        # Save the top_categories to a file
        with open('top_categories.pkl', 'wb') as f:
            pickle.dump(self.top_categories, f)

    def vectorize_business(self, business, selected_features):
        if not self.top_categories and os.path.exists('top_categories.pkl'):
            with open('top_categories.pkl', 'rb') as f:
                self.top_categories = pickle.load(f)
        if not self.top_categories:
            raise ValueError("Top categories not set. Call `build_feature_matrix` first.")
        vector = []
        if 'location' in selected_features:
            vector.append(self.location_vector(business))
        if 'hours' in selected_features:
            vector.append(self.hours_vector(business))
        if 'categories' in selected_features:
            vector.append(self.category_vector(business))
        if 'stars' in selected_features:
            vector.append(self.stars_vector(business))
        if 'sentiment' in selected_features:
            vector.append(self.sentiment_vector(business.get('text', '')))
        return np.hstack(vector)
    
    def build_feature_matrix(self, business_data, selected_features, parallelize=True):
        if not self.top_categories:
            self.save_top_categories(business_data)
        if parallelize:
            feature_matrix = Parallel(n_jobs=-1)(
                delayed(self.vectorize_business)(business, selected_features)
                for _, business in tqdm(business_data.iterrows(), total=business_data.shape[0], desc="Processing Businesses")
            )
        else:
            feature_matrix = []
            for _, business in tqdm(business_data.iterrows(), total=len(business_data), desc="Building feature matrix"):
                vector = self.vectorize_business(business, selected_features)
                feature_matrix.append(vector)

        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        return feature_matrix, feature_matrix_scaled, scaler
    
    def combine_cached_feature_matrices(self, matrix_cache, selected_features):
        feature_matrices = [matrix_cache[k] for k in selected_features]
        combined = np.hstack(feature_matrices)
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)
        return combined, combined_scaled, scaler
    
    def __getstate__(self):
        """
        Customize what gets pickled. Exclude the tokenizer and model.
        """
        state = self.__dict__.copy()
        # Exclude large or non-pickleable objects
        if "sentiment_tokenizer" in state:
            del state["sentiment_tokenizer"]
        if "sentiment_model" in state:
            del state["sentiment_model"]
        return state

    def __setstate__(self, state):
        """
        Customize how the object is restored during unpickling.
        """
        # Restore the state
        self.__dict__.update(state)
        # Reinitialize excluded attributes
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.top_categories = [
            'Restaurants',
            'Food',
            'Shopping',
            'Home Services',
            'Beauty & Spas',
            'Nightlife',
            'Health & Medical',
            'Local Services',
            'Bars',
            'Automotive'
        ]