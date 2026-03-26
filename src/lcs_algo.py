# lcs_algo.py
# Responsible: Rebecca Wang
# LCS dynamic-programming algorithm and similarity utilities.

from typing import List, Dict, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Core LCS algorithm
# ---------------------------------------------------------------------------

def lcs_length(X: List[int], Y: List[int]) -> int:
    """Compute the length of the Longest Common Subsequence of X and Y.

    Uses a space-optimised two-row DP so memory is O(min(m, n)).
    Time complexity: O(m * n).
    """
    if len(X) < len(Y):
        X, Y = Y, X

    m, n = len(X), len(Y)
    if n == 0:
        return 0

    prev = [0] * (n + 1)
    cur  = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev, cur = cur, [0] * (n + 1)

    return prev[n]


def lcs_length_full_table(X: List[int], Y: List[int]) -> np.ndarray:
    """Compute the full LCS DP table (needed for backtracking).

    Recurrence (CLRS 15.4):
        L[i][j] = L[i-1][j-1] + 1          if X[i-1] == Y[j-1]
        L[i][j] = max(L[i-1][j], L[i][j-1]) otherwise

    Returns:
        2-D numpy array L of shape (m+1, n+1).
    """
    m, n = len(X), len(Y)
    L = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L


def recover_lcs(X: List[int], Y: List[int], L: np.ndarray = None) -> List[int]:
    """Backtrack through the DP table to recover the actual LCS.

    Args:
        X, Y: Input sequences.
        L:    Optional pre-computed full DP table; computed if not provided.

    Returns:
        The LCS as a list of elements.
    """
    if L is None:
        L = lcs_length_full_table(X, Y)

    i, j = len(X), len(Y)
    result: List[int] = []

    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            result.append(X[i - 1])
            i -= 1
            j -= 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    result.reverse()
    return result


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------

def normalized_lcs_similarity(X: List[int], Y: List[int]) -> float:
    """Normalised LCS similarity: lcs_length(X,Y) / max(|X|, |Y|).

    Returns a value in [0, 1]. Returns 0.0 for empty sequences.
    """
    if not X or not Y:
        return 0.0
    return lcs_length(X, Y) / max(len(X), len(Y))


def compute_similarity_matrix(
    sequences: Dict[int, List[int]]
) -> Tuple[np.ndarray, List[int]]:
    """Compute the full pairwise similarity matrix for all users.

    Args:
        sequences: {userId: sequence} training sequences.

    Returns:
        sim_matrix: (n x n) numpy array; sim_matrix[i][j] is the normalised
                    LCS similarity between user_ids[i] and user_ids[j].
        user_ids:   Ordered list of user IDs corresponding to matrix rows/cols.
    """
    user_ids = list(sequences.keys())
    n = len(user_ids)
    sim_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        sim_matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = normalized_lcs_similarity(sequences[user_ids[i]], sequences[user_ids[j]])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    return sim_matrix, user_ids


def find_top_k_similar(
    user_id: int,
    user_ids: List[int],
    sim_matrix: np.ndarray,
    k: int = 10,
) -> List[Tuple[int, float]]:
    """Return the top-k most similar users for a given user.

    Args:
        user_id:    Target user ID.
        user_ids:   Ordered list of user IDs (rows/cols of sim_matrix).
        sim_matrix: Pairwise similarity matrix.
        k:          Number of neighbours to return.

    Returns:
        List of (neighbour_user_id, similarity_score) sorted descending.
    """
    idx = user_ids.index(user_id)
    scores = [
        (user_ids[j], float(sim_matrix[idx][j]))
        for j in range(len(user_ids))
        if j != idx
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def compute_all_similarities(
    sequences: Dict[int, List[int]]
) -> Dict[Tuple[int, int], float]:
    """Compute pairwise similarities and return as a flat dict.

    Key: (user_id_a, user_id_b), value: normalised LCS similarity.
    Both (a, b) and (b, a) are stored for convenient lookup.
    """
    sim_matrix, user_ids = compute_similarity_matrix(sequences)
    similarities: Dict[Tuple[int, int], float] = {}

    for i, uid1 in enumerate(user_ids):
        for j, uid2 in enumerate(user_ids):
            if i != j:
                similarities[(uid1, uid2)] = float(sim_matrix[i][j])

    return similarities
