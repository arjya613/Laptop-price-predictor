# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
df = pd.read_csv("C:/Users/HP/OneDrive/Documents/GitHub/Laptop-price-predictor/laptop_data.csv")
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("laptop_data.csv")

# Selecting features for recommendation
features = ['Price', 'Weight', 'Ram', 'SSD', 'HDD']
df_features = df[features]

# Normalize the data
#scaler = StandardScaler()
#scaled_features = scaler.fit_transform(df_features)

# Train KNN model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
#knn.fit(scaled_features)

# Save model, scaler, and dataset
pickle.dump(knn, open('knn_model.pkl', 'wb'))
#pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(df, open('laptop_data.pkl', 'wb'))
