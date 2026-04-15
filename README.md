# Movie Recommender System — Collaborative Filtering

A collaborative filtering recommender system built on the MovieLens dataset, using matrix factorization with learned user and movie embeddings to predict ratings and estimate content value.

---

## Project Overview

Recommender systems are at the core of how platforms like Netflix, Spotify, and Amazon personalize user experiences. This project builds a collaborative filtering model from scratch to predict how users would rate movies they haven't seen yet. Beyond rating prediction, we explore embedding-based movie representations, bias interpretation, PCA visualizations of the learned embedding space, and a novel movie valuation framework that estimates the monetary value of a film based on predicted user engagement.

Key goals of the project:
- Train a matrix factorization model to predict user-movie ratings
- Tune hyperparameters (embedding dimension, learning rate, L2 regularization) to minimize RMSE
- Interpret learned movie and user biases
- Visualize the embedding space using PCA to uncover genre and thematic clusters
- Estimate the economic value of movies using a behavior-driven valuation approach

---

## Repository Structure

```
movie-recommender-collaborative-filtering/
├── README.md
├── Case_2_CollaborativeFiltering_Exercises.ipynb   # Full pipeline notebook
├── Case_2_Team_38.pdf                              # Project report
└── data/                                           # MovieLens dataset (download link below)
    └── ratings.csv / movies.csv
```

---

## Dataset

- **Source:** [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- **Rating Scale:** 1–5 stars
- **Key observations:**
  - Movie popularity is highly skewed — most movies receive fewer than 200 ratings
  - User activity is similarly skewed — most users rate fewer than 100 movies
  - Highest-rated movies (avg 5.0) often have only 1–3 ratings, inflating their scores
  - Well-known classics like *The Shawshank Redemption*, *The Godfather*, and *Schindler's List* are both highly and widely rated

---

## Model — Matrix Factorization (Collaborative Filtering)

The model learns dense vector embeddings for both users and movies, along with user and movie bias terms. The predicted rating for a user-movie pair is computed as:

```
rating = global_mean + user_bias + movie_bias + dot(user_embedding, movie_embedding)
```

- **Loss Function:** Mean Squared Error (MSE)
- **Early Stopping:** Validation loss monitored per epoch; optimal checkpoint at **epoch 5** (before overfitting)

---

## Hyperparameter Tuning

| Embedding Dim | Learning Rate | L2 Reg | RMSE   | MAE    |
|--------------|--------------|--------|--------|--------|
| 32           | 0.01         | 0      | 1.0090 | 0.7760 |
| 32           | 0.01         | 1e-4   | 0.9661 | 0.7663 |
| 32           | 0.05         | 1e-4   | 0.9559 | 0.7585 |
| 64           | 0.01         | 1e-4   | 0.9650 | 0.7655 |
| 64           | 0.005        | 1e-4   | 0.9572 | 0.7595 |

**Best configuration:** `embedding_dim=32, learning_rate=0.05, L2=1e-4`  
**Test RMSE: 0.956 · Test MAE: 0.759** — a meaningful improvement over the default baseline (RMSE 1.009)

---

## Test Results

| Metric    | Value  |
|-----------|--------|
| Test MSE  | 1.0133 |
| Test RMSE | 1.0066 |
| Test MAE  | 0.7745 |

---

## Key Analyses

### Movie Bias Interpretation
The top 10 movies by learned bias (movies rated positively regardless of who rates them) include *American Beauty*, *The Shawshank Redemption*, *Schindler's List*, *The Godfather*, and *Star Wars* — all well-known classics, confirming the model is learning meaningful patterns from the data.

### PCA of Movie Embeddings
Reducing the learned embeddings to 2D via PCA reveals clear genre clustering:
- **Top-right:** Light-hearted comedies and family films (Toy Story, Big, E.T.)
- **Bottom-left:** Dark dramas and thrillers (Pulp Fiction, Fight Club, Goodfellas)
- **Middle-right:** Action and sci-fi (Star Wars, Terminator, The Matrix)
- **Far left:** Classic/older films (Casablanca, Wizard of Oz, 2001: A Space Odyssey)

The x-axis encodes genre (drama/comedy → action/sci-fi), while the y-axis captures emotional tone (uplifting → dark/serious).

### Movie Valuation Framework
A behavior-driven valuation model estimates the monetary value of each movie based on predicted ratings and modeled viewing time using Zipf's Law assumptions:

| Movie | Estimated Value |
|-------|----------------|
| The Godfather (1972) | $47,539,890 |
| The Shawshank Redemption (1994) | $44,332,960 |
| Close Shave, A (1995) | $33,434,210 |
| Usual Suspects, The (1995) | $26,808,600 |
| Schindler's List (1993) | $24,916,990 |
| Toy Story (1995) | $2,278,995 |

This approach captures **depth of engagement** rather than raw popularity, offering a more nuanced complement to view-count-based metrics. Many top-rated-by-volume movies do not appear in the top-valued list, highlighting the difference between breadth of reach and quality of engagement.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-recommender-collaborative-filtering.git
   cd movie-recommender-collaborative-filtering
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn torch jupyter
   ```

3. Download the MovieLens dataset from [grouplens.org](https://grouplens.org/datasets/movielens/) and place it in the `data/` folder.

4. Open and run the notebook:
   ```bash
   jupyter notebook Case_2_CollaborativeFiltering_Exercises.ipynb
   ```

---

## License

Dataset sourced from GroupLens under its respective license.
