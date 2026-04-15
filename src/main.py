# main.py
# Full pipeline: data → LCS similarity → recommendations → evaluation → figures.

import os
import sys
import time

# ── allow imports from src/ when running from repo root ──────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_processing import (
    load_ratings, load_movies,
    build_user_interaction_sequences, train_test_split_interactions,
    get_top_users_by_activity,
)
from lcs_algo import compute_similarity_matrix, recover_lcs
from recommendation import (
    recommend_all_users, random_recommend, evaluate_all_metrics,
)
from visualize_results import (
    plot_hit_rate_comparison, plot_hit_rate_curve,
    plot_similarity_heatmap, plot_similarity_distribution,
    plot_sequence_comparison, plot_metrics_table,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR          = os.path.join(os.path.dirname(__file__), '..', 'ml-latest-small')
MIN_RATING        = 4.0    # treat only high ratings as positive feedback
MAX_USERS         = 150    # top-N active users; more users add noise and runtime
MAX_SEQ_LEN       = 50     # recent history is usually more predictive than long tails
MIN_SEQ_LEN       = 10     # keep enough history to form a stable sequence
TOP_K_USERS       = 5      # collaborative-filtering neighbourhood size
TOP_N_ITEMS       = 20     # recommendation list length
USE_IDF           = False  # niche-item boost hurt next-item accuracy in local tests
SIMILARITY_POWER  = 1.0    # alpha in sim(u, v)^alpha
RECENCY_WEIGHT    = 0.2    # beta in 1 + beta * recency_v(item)
POPULARITY_WEIGHT = 0.05   # lambda for the popularity prior
RATING_WEIGHT     = 0.50   # gamma for neighbour rating strength
K_VALUES          = [1, 5, 10, 20]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    print('LCS-based Movie Recommendation System')
    print('=' * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print('\n[1/5] Loading MovieLens data …')
    ratings_df = load_ratings(os.path.join(DATA_DIR, 'ratings.csv'))
    movies_df  = load_movies(os.path.join(DATA_DIR,  'movies.csv'))
    ratings_df = ratings_df[ratings_df['rating'] >= MIN_RATING].copy()
    ratings_df = ratings_df.sort_values(['userId', 'timestamp'])
    movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
    print(f'      {len(ratings_df):,} ratings, {ratings_df["userId"].nunique()} users, '
          f'{ratings_df["movieId"].nunique()} movies  |  rating >= {MIN_RATING}')

    # ── 2. Build sequences ───────────────────────────────────────────────────
    print('\n[2/5] Building user viewing sequences …')
    all_interactions = build_user_interaction_sequences(
        ratings_df, min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN)
    all_interactions = get_top_users_by_activity(all_interactions, MAX_USERS)
    train_seqs, train_interactions, _test_labels = train_test_split_interactions(all_interactions)
    # Wrap single int labels into lists so evaluation functions support multi-label.
    test_labels = {uid: [item] for uid, item in _test_labels.items()}

    print(f'      {len(train_seqs)} users kept  |  '
          f'avg sequence length: {sum(len(s) for s in train_seqs.values()) / len(train_seqs):.1f}')

    all_items = list({mid for seq in train_seqs.values() for mid in seq})

    # ── 3. Compute LCS similarity matrix ────────────────────────────────────
    print(f'\n[3/5] Computing pairwise LCS similarities ({len(train_seqs)} users) …')
    t0 = time.time()
    sim_matrix, user_ids = compute_similarity_matrix(train_seqs)
    elapsed = time.time() - t0
    print(f'      Done in {elapsed:.1f}s')

    # ── 4. Generate recommendations ──────────────────────────────────────────
    print('\n[4/5] Generating recommendations …')
    lcs_recs    = recommend_all_users(user_ids, sim_matrix, train_seqs,
                                      train_interactions=train_interactions,
                                      top_k_users=TOP_K_USERS,
                                      top_n_items=TOP_N_ITEMS,
                                      use_idf=USE_IDF,
                                      similarity_power=SIMILARITY_POWER,
                                      recency_weight=RECENCY_WEIGHT,
                                      popularity_weight=POPULARITY_WEIGHT,
                                      rating_weight=RATING_WEIGHT,
                                      min_rating_for_weight=MIN_RATING)
    random_recs = random_recommend(train_seqs, all_items, top_n_items=TOP_N_ITEMS)

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    print('\n[5/5] Evaluating …')
    lcs_metrics    = evaluate_all_metrics(lcs_recs,    test_labels, K_VALUES)
    random_metrics = evaluate_all_metrics(random_recs, test_labels, K_VALUES)

    print()
    print(f'  {"K":<6} {"LCS Hit@K":>12} {"Rand Hit@K":>12} {"LCS NDCG@K":>12} {"Rand NDCG@K":>13}')
    print('  ' + '-' * 55)
    for k in K_VALUES:
        lcs_hr   = lcs_metrics['hit_rate'][k]
        rand_hr  = random_metrics['hit_rate'][k]
        lcs_ndcg = lcs_metrics['ndcg'][k]
        rand_ndcg = random_metrics['ndcg'][k]
        print(f'  K={k:<4} {lcs_hr:>12.4f} {rand_hr:>12.4f} {lcs_ndcg:>12.4f} {rand_ndcg:>13.4f}')
    print(f'\n  MRR:  LCS = {lcs_metrics["mrr"]:.4f}  vs  Random = {random_metrics["mrr"]:.4f}')

    # ── Visualisations ───────────────────────────────────────────────────────
    print('\nGenerating figures …')

    plot_hit_rate_comparison(
        lcs_metrics['hit_rate'], random_metrics['hit_rate'], K_VALUES)

    plot_hit_rate_curve(
        lcs_metrics['hit_rate'], random_metrics['hit_rate'], K_VALUES)

    plot_similarity_heatmap(sim_matrix, user_ids, sample_size=30)

    plot_similarity_distribution(sim_matrix)

    plot_metrics_table(lcs_metrics, random_metrics, K_VALUES)

    # Sequence comparison: pick the pair with the highest similarity
    _plot_best_pair_comparison(sim_matrix, user_ids, train_seqs, movie_titles)

    print('\nAll figures saved to figures/')
    print('Done.')


def _plot_best_pair_comparison(sim_matrix, user_ids, train_seqs, movie_titles):
    """Find the most similar user pair and visualise their sequences."""
    import numpy as np
    n = len(user_ids)
    best_i, best_j, best_sim = 0, 1, -1.0
    upper = np.triu_indices(n, k=1)
    flat_idx = int(np.argmax(sim_matrix[upper]))
    best_i = upper[0][flat_idx]
    best_j = upper[1][flat_idx]
    best_sim = sim_matrix[best_i][best_j]

    uid_a, uid_b = user_ids[best_i], user_ids[best_j]
    seq_a, seq_b = train_seqs[uid_a], train_seqs[uid_b]
    lcs_seq = recover_lcs(seq_a, seq_b)

    print(f'  Best pair: User {uid_a} & User {uid_b}  (similarity={best_sim:.3f}, LCS={len(lcs_seq)})')
    plot_sequence_comparison(seq_a, seq_b, lcs_seq, uid_a, uid_b,
                             movie_titles=movie_titles, max_items=15)


if __name__ == '__main__':
    main()
