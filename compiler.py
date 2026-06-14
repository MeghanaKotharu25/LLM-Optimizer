import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

class LLMCompiler:
    def __init__(self, results_file="results.csv"):
        self.df = pd.read_csv(results_file)
        self.encoder = LabelEncoder()
        self.model = self._train_policy()

    def _train_policy(self):
        # Feature Engineering
        df = self.df.copy()
        df['model_encoded'] = self.encoder.fit_transform(df['Model'])
        
        X = df[['model_encoded', 'Size(MB)']]
        y = df['Latency(s)']
        
        regressor = RandomForestRegressor(n_estimators=100)
        regressor.fit(X, y)
        return regressor

    def compile(self, target_latency):
        # Predict which models will be under target_latency
        candidates = self.df.copy()
        candidates['model_encoded'] = self.encoder.transform(candidates['Model'])
        
        predicted_latencies = self.model.predict(candidates[['model_encoded', 'Size(MB)']])
        candidates['predicted_latency'] = predicted_latencies
        
        # Filter by Prediction
        valid = candidates[candidates['predicted_latency'] <= target_latency]
        
        if valid.empty:
            return None, "No models predicted to meet latency target."
            
        # Select best by OES (Optimization Efficiency Score)
        valid['OES'] = 1 / (valid['Perplexity'] * valid['predicted_latency'])
        best = valid.sort_values('OES', ascending=False).iloc[0]
        return best, "Success"