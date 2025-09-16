# imdb_series.py
# Movie Rating Prediction using regression

import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import sparse
import matplotlib.pyplot as plt
import joblib

# ==============================
# Step 1: Load Dataset from ZIP
# ==============================
ZIP_PATH = "IMDb Movies India.csv.zip"  # make sure this file is in same folder

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    csv_name = [f for f in z.namelist() if f.lower().endswith('.csv')][0]
    with z.open(csv_name) as f:
        df = pd.read_csv(f, encoding='latin1')

print("Dataset Loaded:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# ==============================
# Step 2: Identify Target Column
# ==============================
target_col = None
for cand in ['IMDB','IMDB Rating','IMDb','IMDb Rating','rating','Rating','imdb_rating','avg_vote','vote_average']:
    if cand in df.columns:
        target_col = cand
        break

if target_col is None:
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numcols:
        target_col = numcols[0]
    else:
        raise ValueError("No numeric target column found.")

print("Target column:", target_col)
print(df[target_col].describe())

# ==============================
# Step 3: Detect Feature Columns
# ==============================
def find_like(keywords):
    for k in df.columns:
        if any(w.lower() in k.lower() for w in keywords):
            return k
    return None

genre_col = find_like(['genre','genres'])
director_col = find_like(['director'])
actors_col = find_like(['actor','actors','cast','starring','stars'])
title_col = find_like(['title','name','movie'])

use_cols = [c for c in [genre_col, director_col, actors_col, title_col] if c is not None]
print("Using feature columns:", use_cols)

# ==============================
# Step 4: Preprocess Data
# ==============================
df_model = df.dropna(subset=[target_col]).copy()

for c in use_cols:
    df_model[c] = df_model[c].fillna('')

df_model['num_genres'] = df_model[genre_col].astype(str).apply(lambda x: len(x.split(',')) if x else 0) if genre_col else 0
df_model['num_actors'] = df_model[actors_col].astype(str).apply(lambda x: len(x.split(',')) if x else 0) if actors_col else 0

X_base = df_model[use_cols + ['num_genres','num_actors']]
y = df_model[target_col].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

# ==============================
# Step 5: Feature Engineering
# ==============================
mats_train = []
mats_test = []

# Genres & Actors vectorization
for tf in [genre_col, actors_col]:
    if tf and tf in X_base.columns:
        v = CountVectorizer(token_pattern=r"[^,\s]+", max_features=2000)
        v.fit(X_train[tf].astype(str))
        mats_train.append(v.transform(X_train[tf].astype(str)))
        mats_test.append(v.transform(X_test[tf].astype(str)))

# Director One-hot (top 50)
if director_col and director_col in X_base.columns:
    topk = X_train[director_col].fillna('').value_counts().nlargest(50).index.tolist()
    dir_train = X_train[director_col].fillna('').apply(lambda v: v if v in topk else 'OTHER').astype(str)
    dir_test = X_test[director_col].fillna('').apply(lambda v: v if v in topk else 'OTHER').astype(str)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    ohe.fit(dir_train.values.reshape(-1,1))

    mats_train.append(sparse.csr_matrix(ohe.transform(dir_train.values.reshape(-1,1))))
    mats_test.append(sparse.csr_matrix(ohe.transform(dir_test.values.reshape(-1,1))))

#
