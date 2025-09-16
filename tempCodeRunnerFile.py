# Re-attempt reading CSV with latin1 encoding and run a quick model training & evaluation.
import zipfile, os, io, pandas as pd, numpy as np
from caas_jupyter_tools import display_dataframe_to_user

zip_path = "/mnt/data/IMDb Movies India.csv.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    csv_name = [f for f in z.namelist() if f.lower().endswith('.csv')][0]
    with z.open(csv_name) as f:
        df = pd.read_csv(f, encoding='latin1')

print("Read CSV with latin1 encoding. Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Quick preview
display_dataframe_to_user("Movie dataset preview (first 200 rows)", df.head(200))

# Find target column (rating-like)
possible_targets = [c for c in df.columns if 'rating' in c.lower() or 'score' in c.lower() or 'imdb' in c.lower()]
possible_targets, df.select_dtypes(include=[np.number]).columns.tolist()[:10]

# Choose a plausible target: look for columns 'IMDB' or 'Rating'
target_col = None
for cand in ['IMDB','IMDB Rating','IMDb','IMDb Rating','rating','Rating','imdb_rating','avg_vote','vote_average']:
    if cand in df.columns:
        target_col = cand
        break
# fallback to first numeric with many non-null values
if target_col is None:
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numcols:
        target_col = numcols[0]
    else:
        raise ValueError("No numeric target column found.")
print("Selected target column:", target_col)

# Inspect target distribution
print(df[target_col].describe())

# Identify feature columns for genre, director, actors
cols = df.columns.tolist()
def find_like(keywords):
    for k in cols:
        if any(w.lower() in k.lower() for w in keywords):
            return k
    return None

genre_col = find_like(['genre','genres','category'])
director_col = find_like(['director','directed'])
actors_col = find_like(['actor','actors','cast','starring','stars'])
title_col = find_like(['title','name','movie'])

print("Detected columns -> genre:", genre_col, "director:", director_col, "actors:", actors_col, "title:", title_col)

# For modeling, require at least one text-like feature and the target
use_cols = [c for c in [genre_col,director_col,actors_col,title_col] if c is not None]
print("Using feature columns:", use_cols)

# Drop rows with missing target
df_model = df.dropna(subset=[target_col]).copy()
# Fill missing feature text with empty string
for c in use_cols:
    df_model[c] = df_model[c].fillna('')

# Create numeric simple features
if genre_col:
    df_model['num_genres'] = df_model[genre_col].astype(str).apply(lambda x: len(str(x).split(',')) if x else 0)
else:
    df_model['num_genres'] = 0
if actors_col:
    df_model['num_actors'] = df_model[actors_col].astype(str).apply(lambda x: len(str(x).split(',')) if x else 0)
else:
    df_model['num_actors'] = 0

# Prepare feature matrix: vectorize genres and actors, one-hot top directors
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import sparse

X_base = df_model[use_cols + ['num_genres','num_actors']]
y = df_model[target_col].astype(float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

# Vectorizers for text fields
vecs = {}
mats_train = []
mats_test = []

for tf in [genre_col, actors_col]:
    if tf and tf in X_base.columns:
        v = CountVectorizer(token_pattern=r"[^,\s]+", max_features=2000)
        v.fit(X_train[tf].astype(str))
        mats_train.append(v.transform(X_train[tf].astype(str)))
        mats_test.append(v.transform(X_test[tf].astype(str)))

# Director one-hot (top 50)
if director_col and director_col in X_base.columns:
    topk = X_train[director_col].fillna('').value_counts().nlargest(50).index.tolist()
    dir_train = X_train[director_col].fillna('').apply(lambda v: v if v in topk else 'OTHER').astype(str)
    dir_test = X_test[director_col].fillna('').apply(lambda v: v if v in topk else 'OTHER').astype(str)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ohe.fit(dir_train.values.reshape(-1,1))
    mats_train.append(sparse.csr_matrix(ohe.transform(dir_train.values.reshape(-1,1))))
    mats_test.append(sparse.csr_matrix(ohe.transform(dir_test.values.reshape(-1,1))))

# Numeric features
num_train = X_train[['num_genres','num_actors']].values
num_test = X_test[['num_genres','num_actors']].values
mats_train.insert(0, sparse.csr_matrix(num_train))
mats_test.insert(0, sparse.csr_matrix(num_test))

# Combine
X_train_final = sparse.hstack(mats_train).tocsr()
X_test_final = sparse.hstack(mats_test).tocsr()
print("Final feature matrix shapes:", X_train_final.shape, X_test_final.shape)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# Show a few predictions vs actual
res_df = X_test.reset_index(drop=True).copy()
res_df['actual'] = y_test.reset_index(drop=True)
res_df['predicted'] = y_pred
display_dataframe_to_user("Sample predictions (test set)", res_df[[c for c in use_cols if c in res_df.columns]+['num_genres','num_actors','actual','predicted']].head(200))

# Plot actual vs predicted (matplotlib, single plot)
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, s=10)
plt.xlabel("Actual "+str(target_col))
plt.ylabel("Predicted")
plt.title("Actual vs Predicted - Movie Ratings")
plt.grid(True)
plt.show()

# Save simple model
import joblib
joblib.dump(model, "/mnt/data/movie_rating_model_notebook_rf.joblib")
print("Saved model to /mnt/data/movie_rating_model_notebook_rf.joblib")
