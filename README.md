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

- Only ratings ≥ 3.5 are kept as positive interactions
- Sequences are constructed chronologically using timestamps
- Users with fewer than 5 interactions are excluded
- The last item in each sequence is held out for evaluation

---

## Project Structure

```
LCS/
├── ml-latest-small/        # MovieLens dataset
│   ├── ratings.csv
│   ├── movies.csv
│   ├── links.csv
│   └── tags.csv
├── src/
│   ├── data_processing.py  # Data loading, cleaning, sequence construction
│   ├── lcs_algo.py         # Core LCS dynamic programming algorithm
│   ├── recommendation.py   # Recommendation logic and evaluation metrics
│   ├── visualize_results.py# Result visualizations and plots
│   └── main.py             # End-to-end pipeline
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
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

---

## Evaluation

Prediction performance is compared between:
- **LCS-based recommendations** — uses sequence similarity to predict next item
- **Random baseline** — recommends a random movie

Metrics used: Hit Rate, MRR, Precision@N, Recall@N

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