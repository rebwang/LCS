# Sequence-Based Behavioral Recommendation Using Longest Common Subsequence

**CS 5800 Final Project — Group 10**  
Northeastern University

---

## Overview

This project explores how user viewing histories can be represented as sequences and compared using the Longest Common Subsequence (LCS) algorithm to build a recommendation system. Rather than relying on machine learning models, we use a classical dynamic programming approach to identify users with similar behavioral patterns and predict their next content choice.

---

## Research Question

Can sequence similarity measured by the LCS algorithm accurately predict a user's next content choice?

---

## How It Works

1. User viewing histories are represented as ordered sequences of movie IDs
2. The LCS algorithm computes similarity between pairs of user sequences
3. Users with the most similar sequences are identified as "neighbors"
4. The next item is predicted based on what similar users watched
5. Predictions are evaluated against a held-out test item per user

### LCS Recurrence

```
If X[i] == Y[j]:   L[i][j] = L[i-1][j-1] + 1
Else:              L[i][j] = max(L[i-1][j], L[i][j-1])
```

Time complexity: **O(mn)** where m and n are sequence lengths.

---

## Dataset

We use the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) dataset, which contains 100,836 ratings across 9,742 movies from 610 users.

- Only ratings ≥ 4.0 are used to construct viewing sequences (positive-feedback filter)
- Sequences are sorted chronologically using timestamps
- Users with fewer than 10 rated movies are excluded
- Sequences are truncated to the most recent 50 items per user
- Top 100 most active users are selected for evaluation
- The last item in each sequence is held out for evaluation (leave-one-out)

---

## Project Structure

```
LCS/
├── ml-latest-small/         # MovieLens dataset
│   ├── ratings.csv
│   ├── movies.csv
│   ├── links.csv
│   └── tags.csv
├── src/
│   ├── data_processing.py   # Data loading, cleaning, sequence construction
│   ├── lcs_algo.py          # Core LCS dynamic programming algorithm
│   ├── recommendation.py    # Recommendation logic and evaluation metrics
│   ├── visualize_results.py # Result visualizations and plots
│   ├── main.py              # End-to-end pipeline
│   └── app.py               # Streamlit interactive frontend
├── figures/                 # Auto-generated output figures
├── tests/
│   └── test_lcs.py          # Unit tests for LCS algorithm and similarity
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/rebwang/LCS.git
cd LCS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python src/main.py
```

Output includes:
- Data loading stats and sequence counts
- Pairwise LCS similarity computation
- Full metric evaluation (LCS vs random baseline): Hit Rate, NDCG, MRR, Precision, Recall
- 6 figures saved to the `figures/` folder

### 4. Run the Streamlit app (optional)
```bash
streamlit run src/app.py
```

This launches an interactive web interface where you can:
- Browse recommendations for any user
- Explore the user-user similarity heatmap
- Visualize the LCS DP table for any two users
- Compare LCS vs random evaluation metrics

### 5. Run the tests
```bash
pytest tests/test_lcs.py -v
```

Tests cover: `lcs_length`, `lcs_length_full_table`, `recover_lcs`, `normalized_lcs_similarity`, `compute_similarity_matrix`, and `find_top_k_similar`. All 13 tests pass.

---

## Evaluation

Prediction performance is compared between:
- **LCS-based recommendations** — uses sequence similarity to predict next item
- **Random baseline** — recommends a random movie

Metrics used: Hit Rate, MRR, Precision@N, Recall@N, and NDCG@N evaluated at N ∈ {1, 5, 10, 20}

---

## Team

| Name | Role |
|------|------|
| Soonbee Hwang | Data Processing & Sequence Construction |
| Rebecca Wang | Algorithm Implementation & Similarity Computation |
| Yue Liang | Recommendation System & Results Analysis |

---

## Reference

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4.