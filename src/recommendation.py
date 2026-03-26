# recommendation.py
# Responsible: Yue Liang
# Generates recommendations from LCS-based user similarity and evaluates them.

import random
from typing import Dict, List, Tuple
import numpy as np

from lcs_algo import find_top_k_similar


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def recommend_for_user(
    user_id: int,
    user_ids: List[int],
    sim_matrix: np.ndarray,
    train_sequences: Dict[int, List[int]],
    top_k_users: int = 10,
    top_n_items: int = 10,
) -> List[int]:
    """Generate Top-N item recommendations for a single user.

    Strategy (user-based collaborative filtering via LCS similarity):
      1. Find the top_k_users most similar neighbours.
      2. Collect items that each neighbour watched but the target user has not.
      3. Weight each candidate item by the neighbour's similarity score.
      4. Return the top_n_items items ranked by accumulated score.

    Args:
        user_id:         The target user to generate recommendations for.
        user_ids:        Ordered list of all user IDs (matches sim_matrix axes).
        sim_matrix:      Pre-computed pairwise similarity matrix.
        train_sequences: {userId: [movieId, ...]} training sequences.
        top_k_users:     Number of nearest neighbours to consider.
        top_n_items:     Number of items to recommend.

    Returns:
        List of recommended movieIds, ranked by predicted relevance.
    """
    neighbours = find_top_k_similar(user_id, user_ids, sim_matrix, k=top_k_users)

    watched = set(train_sequences.get(user_id, []))
    item_scores: Dict[int, float] = {}

    for neighbour_id, sim_score in neighbours:
        if sim_score <= 0:
            continue
        for item in train_sequences.get(neighbour_id, []):
            if item not in watched:
                item_scores[item] = item_scores.get(item, 0.0) + sim_score

    ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_n_items]]


def recommend_all_users(
    user_ids: List[int],
    sim_matrix: np.ndarray,
    train_sequences: Dict[int, List[int]],
    top_k_users: int = 10,
    top_n_items: int = 20,
) -> Dict[int, List[int]]:
    """Generate recommendations for every user in train_sequences.

    Returns:
        {userId: [recommended_movieId, ...]}
    """
    recommendations: Dict[int, List[int]] = {}
    for uid in train_sequences:
        recommendations[uid] = recommend_for_user(
            uid, user_ids, sim_matrix, train_sequences,
            top_k_users=top_k_users, top_n_items=top_n_items,
        )
    return recommendations


def random_recommend(
    train_sequences: Dict[int, List[int]],
    all_items: List[int],
    top_n_items: int = 20,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Random-baseline recommender: samples unseen items uniformly at random.

    Args:
        train_sequences: Used to determine which items each user has already seen.
        all_items:       Pool of all known item IDs.
        top_n_items:     Number of items to recommend per user.
        seed:            Random seed for reproducibility.

    Returns:
        {userId: [randomly_recommended_movieId, ...]}
    """
    rng = random.Random(seed)
    recommendations: Dict[int, List[int]] = {}

    for uid, seq in train_sequences.items():
        watched = set(seq)
        candidates = [item for item in all_items if item not in watched]
        n = min(top_n_items, len(candidates))
        recommendations[uid] = rng.sample(candidates, n)

    return recommendations


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def hit_rate_at_k(
    recommendations: Dict[int, List[int]],
    test_labels: Dict[int, int],
    k: int,
) -> float:
    """Hit Rate@K: fraction of users for whom the ground-truth next item
    appears in the top-K recommendations.

    Args:
        recommendations: {userId: [ranked_movieId, ...]}
        test_labels:     {userId: ground_truth_movieId}
        k:               Cut-off rank.

    Returns:
        Hit Rate@K in [0, 1].
    """
    hits = 0
    total = 0
    for uid, predicted in recommendations.items():
        if uid not in test_labels:
            continue
        total += 1
        if test_labels[uid] in predicted[:k]:
            hits += 1
    return hits / total if total > 0 else 0.0


def precision_at_k(
    recommendations: Dict[int, List[int]],
    test_labels: Dict[int, int],
    k: int,
) -> float:
    """Precision@K: average fraction of top-K recommendations that are relevant.

    (With a single ground-truth item per user, this equals Hit Rate@K / K.)
    """
    total_precision = 0.0
    total = 0
    for uid, predicted in recommendations.items():
        if uid not in test_labels:
            continue
        total += 1
        hits = 1 if test_labels[uid] in predicted[:k] else 0
        total_precision += hits / k
    return total_precision / total if total > 0 else 0.0


def recall_at_k(
    recommendations: Dict[int, List[int]],
    test_labels: Dict[int, int],
    k: int,
) -> float:
    """Recall@K: fraction of relevant items retrieved in top-K.

    (With one relevant item per user, Recall@K == Hit Rate@K.)
    """
    return hit_rate_at_k(recommendations, test_labels, k)


def evaluate_all_metrics(
    recommendations: Dict[int, List[int]],
    test_labels: Dict[int, int],
    k_values: List[int],
) -> Dict[str, Dict[int, float]]:
    """Compute Hit Rate, Precision, and Recall at each K in k_values.

    Returns:
        {
          'hit_rate':  {k: value, ...},
          'precision': {k: value, ...},
          'recall':    {k: value, ...},
        }
    """
    results = {'hit_rate': {}, 'precision': {}, 'recall': {}}
    for k in k_values:
        results['hit_rate'][k]  = hit_rate_at_k(recommendations, test_labels, k)
        results['precision'][k] = precision_at_k(recommendations, test_labels, k)
        results['recall'][k]    = recall_at_k(recommendations, test_labels, k)
    return results
