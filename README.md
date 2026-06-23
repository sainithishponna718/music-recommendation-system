# Music-Recommender-System
 
## Overview
This project implements a content-based music recommendation system using Spotify song metadata. The recommender suggests songs similar to a given track by leveraging song tags, artist information, audio features, and release year.
The system combines feature engineering, text vectorization, and similarity-based retrieval to generate personalized song recommendations.
 
---
 
## Dataset
The dataset is the [Million Song Dataset + Spotify + Last.fm](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm) (~50,000 tracks), containing Spotify song metadata, including:
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
- Artist analysis (8,317 unique artists)
- Tag analysis
- Distribution analysis of audio features
- Data quality checks
### 2. Data Preprocessing
- Removing duplicates
- Dropping the `genre` column (>50% missing) in favor of the more complete `tags` column
- Filling missing tags with `"no_tags"`
- Normalizing artist names to lowercase
- Feature selection
### 3. Feature Engineering
Each column was encoded according to its type:
 
#### Tags
Tags are transformed using **TF-IDF Vectorization** (max 85 features, selected by requiring a minimum value-count threshold of 1,000) to capture semantic relationships between songs.
 
#### Artist Information
Artist names are encoded using **One-Hot Encoding** (8,317 columns).
 
#### Key & Time Signature
Encoded using **One-Hot Encoding** (12 and 6 unique values respectively).
 
#### Year
Release year is **frequency (count) encoded** to incorporate temporal information.
 
#### Audio Features
- `duration_ms`, `loudness`, `tempo` → **StandardScaler** (unbounded continuous ranges)
- `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` → **MinMaxScaler** (already bounded 0–1)
### 4. Recommendation Engine
The transformed feature vectors are combined into a unified feature space (~8,431 columns total).
Cosine Similarity is then used to measure similarity between songs.
 
For a given song:
1. Retrieve its feature vector
2. Compute cosine similarity with all songs
3. Rank songs by similarity score
4. Return the top-k recommendations
```python
recommend(song_name="Demons", songs_data=df_songs, transformed_data=transformed_df, k=10)
```
 
---
 
## Evaluation
 
The recommendation system was first checked qualitatively on a couple of example songs, then evaluated quantitatively across a random sample of 300 query songs (k=10 recommendations each), benchmarked against a random-recommendation baseline.
 
### Quantitative Results
 
| Metric | Content-Based Model | Random Baseline | What it measures |
|---|---|---|---|
| **Tag overlap** (Jaccard similarity) | **0.41** | 0.03 | Overlap between a recommendation's tags and the query song's tags |
| **Audio similarity** (cosine, scaled features) | **0.85** | 0.04 | Closeness of a recommendation's audio profile (danceability, energy, valence, tempo) to the query song |
| **Artist diversity** | 0.75 | 1.00 | Fraction of recommendations from a *different* artist than the query |
 
**Tag overlap and audio similarity** both show a large lift over chance (~14x and ~21x respectively), confirming the recommender is genuinely surfacing content-similar songs rather than arbitrary ones — not just plausible-looking on a couple of hand-picked examples.
 
**Artist diversity is lower than the random baseline by design.** With 8,317 unique artists in the dataset, two randomly chosen songs are almost never by the same artist, so the random baseline sits near 1.0. The model's 0.75 reflects that `artist` is one-hot-encoded as a feature, so artist identity meaningfully shapes recommendations (~25% share the query's artist). This is a deliberate tradeoff in the current feature set — fans of an artist's catalog will see more of that artist recommended back to them — but it does mean the system leans partly on artist identity, not purely on audio/tag content.
 
### Tag Similarity Analysis
Recommended songs were compared against the query song using tag frequency distributions. The analysis showed strong overlap between dominant tags of the original song and the recommended songs, consistent with the Jaccard results above.
 
### Audio Feature Comparison
The following audio features were compared between the original song and the mean values of recommended songs:
- Danceability
- Energy
- Valence
- Tempo
Results demonstrated that recommended songs generally exhibit similar musical characteristics and emotional profiles to the original song.
 
### Visualization
Evaluation visualizations were created to compare:
- Original song audio features vs. mean audio features of recommended songs
- Tag frequency distributions among recommendations
- Distribution of all three metrics (tag overlap, artist diversity, audio similarity) across the 300-song sample
- Content-based model vs. random baseline, side by side
---
 
## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- category_encoders (Count Encoder)
- kagglehub
---
 
## Key Techniques
- Content-Based Filtering
- TF-IDF Vectorization
- One-Hot Encoding
- Frequency (Count) Encoding
- Feature Scaling (StandardScaler, MinMaxScaler)
- Cosine Similarity
- Random-Baseline Benchmarking
---
 
## Known Limitations
- **Artist dominance**: the one-hot-encoded `artist` feature (8,317 columns) is a large portion of the feature space and visibly shapes recommendations — see Evaluation above.
- **Duplicate song names**: `recommend()` looks up a song by exact name match and takes the first matching row; if two different songs share a name, the wrong one could be selected.
- **No deployment/demo UI yet** — `recommend()` is demo-ready but not yet wrapped in an interactive interface.
---
 
## Future Improvements
- Re-run the evaluation with `artist` excluded from the feature matrix, to quantify how much of the tag/audio lift comes from artist clustering vs. genuine content similarity
- Implement Collaborative Filtering using user listening history
- Build a Hybrid Recommendation System
- Deploy the system as an interactive web application (Gradio/Streamlit)
---
 
## Results
The recommender successfully identifies songs with similar tags and audio characteristics, producing meaningful music recommendations through content-based filtering — with quantitative evidence (14x lift on tag overlap, 21x lift on audio similarity over a random baseline) backing that claim rather than relying on a couple of hand-picked examples alone.
