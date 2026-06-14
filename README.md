# Music-Recommender-System

## Overview

This project implements a content-based music recommendation system using Spotify song metadata. The recommender suggests songs similar to a given track by leveraging song tags, artist information, audio features, and release year.

The system combines feature engineering, text vectorization, and similarity-based retrieval to generate personalized song recommendations.

---

## Dataset

The dataset contains Spotify song metadata, including:

- Song Name
- Artist
- Tags
- Release Year
- Audio Features:
  - Danceability
  - Energy
  - Valence
  - Tempo
  - Acousticness
  - Liveness
  - Instrumentalness
  - Speechiness
  - Loudness
  - Duration
  - Key
  - Time Signature

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Missing value analysis
- Duplicate detection
- Artist analysis
- Tag analysis
- Distribution analysis of audio features
- Data quality checks

### 2. Data Preprocessing

- Handling missing values
- Feature selection
- Frequency encoding for year
- One-hot encoding for categorical features
- Scaling numerical features

### 3. Feature Engineering

#### Tags

Tags are transformed using TF-IDF Vectorization to capture semantic relationships between songs.

#### Artist Information

Artist information is encoded using One-Hot Encoding.

#### Audio Features

Numerical audio features are standardized using StandardScaler.

#### Year

Release year is frequency encoded to incorporate temporal information.

### 4. Recommendation Engine

The transformed feature vectors are combined into a unified feature space.

Cosine Similarity is then used to measure similarity between songs.

For a given song:

1. Retrieve its feature vector
2. Compute cosine similarity with all songs
3. Rank songs by similarity score
4. Return the top-k recommendations

---

## Evaluation

The recommendation system was evaluated using qualitative analysis.

### Tag Similarity Analysis

Recommended songs were compared against the query song using tag frequency distributions.

The analysis showed strong overlap between dominant tags of the original song and the recommended songs, indicating that the recommender successfully captures semantic and stylistic similarity.

### Audio Feature Comparison

The following audio features were compared between the original song and the mean values of recommended songs:

- Danceability
- Energy
- Valence
- Tempo

Results demonstrated that recommended songs generally exhibit similar musical characteristics and emotional profiles to the original song.

### Visualization

Evaluation visualizations were created to compare:

- Original song audio features
- Mean audio features of recommended songs
- Tag frequency distributions among recommendations

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## Key Techniques

- Content-Based Filtering
- TF-IDF Vectorization
- One-Hot Encoding
- Frequency Encoding
- Feature Scaling
- Cosine Similarity

---

## Future Improvements

- Implement Collaborative Filtering using user listening history
- Build a Hybrid Recommendation System
- Evaluate recommendations using Hit Rate@K and Precision@K
- Deploy the system as an interactive web application


---

## Results

The recommender successfully identifies songs with similar tags, artists, and audio characteristics, producing meaningful music recommendations through content-based filtering techniques.
