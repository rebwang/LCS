# recommendation.py
# Responsible: Yue Liang
# Generates recommendations from LCS-based user similarity and evaluates them.

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np

from lcs_algo import find_top_k_similar


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def compute_idf(train_sequences: Dict[int, List[int]]) -> Dict[int, float]:
    """Compute IDF weights for all items in the training corpus.

    idf[item] = log(N / df(item)), where N is the number of users and
    df(item) is the number of users who have watched that item.
    High-IDF items are niche (penalises popular blockbusters naturally).
    """
    N = len(train_sequences)
    df: Dict[int, int] = {}
    for seq in train_sequences.values():
        for item in set(seq):
            df[item] = df.get(item, 0) + 1
    return {item: math.log(N / count) for item, count in df.items()}


def compute_item_popularity(train_sequences: Dict[int, List[int]]) -> Counter:
    """Count how often each item appears in the training sequences."""
    return Counter(item for seq in train_sequences.values() for item in seq)


def compute_normalized_popularity(
    item_popularity: Counter,
) -> Dict[int, float]:
    """Map raw popularity counts to [0, 1] using log scaling."""
    if not item_popularity:
        return {}
    max_count = max(item_popularity.values())
    scale = math.log1p(max_count)
    return {
        item: math.log1p(count) / scale
        for item, count in item_popularity.items()
    }


def recommend_for_user(
    user_id: int,
    user_ids: List[int],
    sim_matrix: np.ndarray,
    train_sequences: Dict[int, List[int]],
    item_popularity: Counter,
    popularity_scores: Dict[int, float],
    idf: Optional[Dict[int, float]] = None,
    train_interactions: Optional[Dict[int, List[Tuple[int, float]]]] = None,
    top_k_users: int = 10,
    top_n_items: int = 10,
    similarity_power: float = 1.0,
    recency_weight: float = 0.2,
    popularity_weight: float = 0.05,
    rating_weight: float = 0.3,
    min_rating_for_weight: float = 0.0,
    max_rating_for_weight: float = 5.0,
) -> List[int]:
    """Generate Top-N item recommendations for a single user.

    Strategy (user-based collaborative filtering via LCS similarity):
      1. Find the top_k_users most similar neighbours.
      2. Collect items that each neighbour watched but the target user has not.
      3. Weight each candidate by:
         sim(u, v)^alpha * (1 + beta * recency_v(item)).
      4. Normalise the collaborative score and add a small popularity prior.
      5. Back off to popularity if neighbours contribute no unseen candidates.
      6. Return the top_n_items items ranked by accumulated score.

    Args:
        user_id:         The target user to generate recommendations for.
        user_ids:        Ordered list of all user IDs (matches sim_matrix axes).
        sim_matrix:      Pre-computed pairwise similarity matrix.
        train_sequences: {userId: [movieId, ...]} training sequences.
        item_popularity: Global item counts for tie-breaking / fallback.
        popularity_scores: Log-normalised popularity prior in [0, 1].
        idf:             Optional IDF weights from compute_idf().
        train_interactions: Optional aligned (movieId, rating) histories.
        top_k_users:     Number of nearest neighbours to consider.
        top_n_items:     Number of items to recommend.

    Returns:
        List of recommended movieIds, ranked by predicted relevance.
    """
    neighbours = find_top_k_similar(user_id, user_ids, sim_matrix, k=top_k_users)

    watched = set(train_sequences.get(user_id, []))
    item_scores: Dict[int, float] = {}
    total_neighbour_weight = 0.0

    for neighbour_id, sim_score in neighbours:
        if sim_score <= 0:
            continue
        sim_weight = sim_score ** similarity_power
        total_neighbour_weight += sim_weight
        if train_interactions is not None:
            neighbour_seq = train_interactions.get(neighbour_id, [])
        else:
            neighbour_seq = [
                (item, max_rating_for_weight)
                for item in train_sequences.get(neighbour_id, [])
            ]
        neighbour_len = max(len(neighbour_seq), 1)

        for pos, (item, rating) in enumerate(neighbour_seq):
            if item not in watched:
                recency_ratio = (pos + 1) / neighbour_len
                if max_rating_for_weight > min_rating_for_weight:
                    normalized_rating = (
                        max(min(rating, max_rating_for_weight), min_rating_for_weight)
                        - min_rating_for_weight
                    ) / (max_rating_for_weight - min_rating_for_weight)
                else:
                    normalized_rating = 0.0

                score = sim_weight * (1.0 + recency_weight * recency_ratio)
                score *= 1.0 + rating_weight * normalized_rating
                if idf is not None:
                    score *= idf.get(item, 1.0)
                item_scores[item] = item_scores.get(item, 0.0) + score

    if not item_scores:
        return [
            item for item, _ in item_popularity.most_common()
            if item not in watched
        ][:top_n_items]

    normalizer = total_neighbour_weight if total_neighbour_weight > 0 else 1.0
    final_scores = {
        item: (score / normalizer) + popularity_weight * popularity_scores.get(item, 0.0)
        for item, score in item_scores.items()
    }
    ranked = sorted(
        final_scores.items(),
        key=lambda x: (x[1], item_popularity.get(x[0], 0)),
        reverse=True,
    )
    return [item for item, _ in ranked[:top_n_items]]


def recommend_all_users(
    user_ids: List[int],
    sim_matrix: np.ndarray,
    train_sequences: Dict[int, List[int]],
    train_interactions: Optional[Dict[int, List[Tuple[int, float]]]] = None,
    top_k_users: int = 10,
    top_n_items: int = 20,
    use_idf: bool = False,
    similarity_power: float = 1.0,
    recency_weight: float = 0.2,
    popularity_weight: float = 0.05,
    rating_weight: float = 0.3,
    min_rating_for_weight: float = 0.0,
    max_rating_for_weight: float = 5.0,
) -> Dict[int, List[int]]:
    """Generate recommendations for every user in train_sequences.

    Computes shared corpus statistics once and reuses them for all users.

    Returns:
        {userId: [recommended_movieId, ...]}
    """
    item_popularity = compute_item_popularity(train_sequences)
    popularity_scores = compute_normalized_popularity(item_popularity)
    idf = compute_idf(train_sequences) if use_idf else None
    recommendations: Dict[int, List[int]] = {}
    for uid in train_sequences:
        recommendations[uid] = recommend_for_user(
            uid, user_ids, sim_matrix, train_sequences,
            item_popularity, popularity_scores, idf, train_interactions,
            top_k_users=top_k_users, top_n_items=top_n_items,
            similarity_power=similarity_power,
            recency_weight=recency_weight,
            popularity_weight=popularity_weight,
            rating_weight=rating_weight,
            min_rating_for_weight=min_rating_for_weight,
            max_rating_for_weight=max_rating_for_weight,
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
    test_labels: Dict[int, List[int]],
    k: int,
) -> float:
    """Hit Rate@K: fraction of users for whom at least one ground-truth item
    appears in the top-K recommendations.

    Args:
        recommendations: {userId: [ranked_movieId, ...]}
        test_labels:     {userId: [ground_truth_movieId, ...]}
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
        relevant = set(test_labels[uid])
        if any(item in relevant for item in predicted[:k]):
            hits += 1
    return hits / total if total > 0 else 0.0


def precision_at_k(
    recommendations: Dict[int, List[int]],
    test_labels: Dict[int, List[int]],
    k: int,
) -> float:
    """Precision@K: average fraction of top-K recommendations that are relevant."""
    total_precision = 0.0
    total = 0
    for uid, predicted in recommendations.items():
        if uid not in test_labels:
            continue
        total += 1
        relevant = set(test_labels[uid])
        hits = sum(1 for item in predicted[:k] if item in relevant)
        total_precision += hits / k
    return total_precision / total if total > 0 else 0.0


def recall_at_k(
    recommendations: Dict[int, List[int]],
    test_labels: Dict[int, List[int]],
    k: int,
) -> float:
    """Recall@K: fraction of all relevant items that appear in the top-K list."""
    total_recall = 0.0
    total = 0
    for uid, predicted in recommendations.items():
        if uid not in test_labels:
            continue
        total += 1
        relevant = set(test_labels[uid])
        hits = sum(1 for item in predicted[:k] if item in relevant)
        total_recall += hits / len(relevant)
    return total_recall / total if total > 0 else 0.0


def mrr(
    recommendations: Dict[int, List[int]],
    test_labels: Dict[int, List[int]],
) -> float:
    """Mean Reciprocal Rank: average of 1/rank of the first relevant hit.

    Rank is 1-indexed; users with no hit contribute 0.

    Args:
        recommendations: {userId: [ranked_movieId, ...]}
        test_labels:     {userId: [ground_truth_movieId, ...]}

    Returns:
        MRR in [0, 1].
    """
    total_rr = 0.0
    total = 0
    for uid, predicted in recommendations.items():
        if uid not in test_labels:
            continue
        total += 1
        relevant = set(test_labels[uid])
        for rank, item in enumerate(predicted, start=1):
            if item in relevant:
                total_rr += 1.0 / rank
                break
    return total_rr / total if total > 0 else 0.0


def ndcg_at_k(
    recommendations: Dict[int, List[int]],
    test_labels: Dict[int, List[int]],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain @ K.

    Items get a gain of 1 if they are relevant, 0 otherwise. The discount
    is log2(rank + 1) so earlier positions matter more. NDCG normalises
    DCG by the ideal DCG (all relevant items ranked first).

    Args:
        recommendations: {userId: [ranked_movieId, ...]}
        test_labels:     {userId: [ground_truth_movieId, ...]}
        k:               Cut-off rank.

    Returns:
        NDCG@K in [0, 1].
    """
    total_ndcg = 0.0
    total = 0
    for uid, predicted in recommendations.items():
        if uid not in test_labels:
            continue
        total += 1
        relevant = set(test_labels[uid])

        # DCG: sum of 1/log2(rank+1) for each relevant item in top-K
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, item in enumerate(predicted[:k], start=1)
            if item in relevant
        )

        # IDCG: best possible DCG — all relevant items ranked first
        n_relevant_in_k = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_relevant_in_k + 1))

        total_ndcg += dcg / idcg if idcg > 0 else 0.0
    return total_ndcg / total if total > 0 else 0.0


def evaluate_all_metrics(
    recommendations: Dict[int, List[int]],
    test_labels: Dict[int, List[int]],
    k_values: List[int],
) -> Dict[str, object]:
    """Compute Hit Rate, MRR, Precision, Recall, and NDCG at each K in k_values.

    Returns:
        {
          'hit_rate':  {k: value, ...},
          'mrr':       float,
          'precision': {k: value, ...},
          'recall':    {k: value, ...},
          'ndcg':      {k: value, ...},
        }
    """
    results: Dict[str, object] = {
        'hit_rate': {}, 'mrr': 0.0, 'precision': {}, 'recall': {}, 'ndcg': {},
    }
    results['mrr'] = mrr(recommendations, test_labels)
    for k in k_values:
        results['hit_rate'][k]  = hit_rate_at_k(recommendations, test_labels, k)
        results['precision'][k] = precision_at_k(recommendations, test_labels, k)
        results['recall'][k]    = recall_at_k(recommendations, test_labels, k)
        results['ndcg'][k]      = ndcg_at_k(recommendations, test_labels, k)
    return results


# ---------------------------------------------------------------------------
# Synthetic smoke-test (run: python recommendation.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from lcs_algo import compute_similarity_matrix

    train = {1: [10, 20, 30, 40], 2: [10, 20, 30, 50, 60], 3: [10, 30, 40, 70]}
    test  = {1: [50, 60], 2: [70, 40], 3: [60, 50]}

    all_items = list({item for seq in train.values() for item in seq})
    sim_matrix, user_ids = compute_similarity_matrix(train)

    lcs_recs    = recommend_all_users(user_ids, sim_matrix, train, top_k_users=2, top_n_items=2)
    random_recs = random_recommend(train, all_items, top_n_items=2, seed=42)

    k_values = [1, 5, 10]
    lcs_results    = evaluate_all_metrics(lcs_recs,    test, k_values)
    random_results = evaluate_all_metrics(random_recs, test, k_values)

    print("Synthetic smoke test (3 users, not a quality benchmark)")
    print(f"  Train: {train}")
    print(f"  Test:  {test}")
    print()
    print(f"  Hit Rate@1:   LCS = {lcs_results['hit_rate'][1]:.4f}  vs  Random = {random_results['hit_rate'][1]:.4f}")
    print(f"  Hit Rate@5:   LCS = {lcs_results['hit_rate'][5]:.4f}  vs  Random = {random_results['hit_rate'][5]:.4f}")
    print(f"  Hit Rate@10:  LCS = {lcs_results['hit_rate'][10]:.4f}  vs  Random = {random_results['hit_rate'][10]:.4f}")
    print(f"  MRR:          LCS = {lcs_results['mrr']:.4f}  vs  Random = {random_results['mrr']:.4f}")
    print(f"  Precision@5:  LCS = {lcs_results['precision'][5]:.4f}  vs  Random = {random_results['precision'][5]:.4f}")
    print(f"  Precision@10: LCS = {lcs_results['precision'][10]:.4f}  vs  Random = {random_results['precision'][10]:.4f}")
    print(f"  Recall@5:     LCS = {lcs_results['recall'][5]:.4f}  vs  Random = {random_results['recall'][5]:.4f}")
    print(f"  Recall@10:    LCS = {lcs_results['recall'][10]:.4f}  vs  Random = {random_results['recall'][10]:.4f}")
    print(f"  NDCG@5:       LCS = {lcs_results['ndcg'][5]:.4f}  vs  Random = {random_results['ndcg'][5]:.4f}")
    print(f"  NDCG@10:      LCS = {lcs_results['ndcg'][10]:.4f}  vs  Random = {random_results['ndcg'][10]:.4f}")
