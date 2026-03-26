from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

def lcs_length(X: List[int], Y: List[int]) -> int:
    """
    Compute the length of the Longest Common Subsequence of X and Y.
    """
    if len(X) < len(Y):
        X, Y = Y, X

    m, n = len(X), len(Y)
    if n == 0:
        return 0
    
    # space optimized dp
    prev = [0] * (n + 1)
    cur = [0] * (n + 1)

    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                cur[j] = prev[j-1] + 1
            else:
                cur[j] = max(prev[j], cur[j-1])
        
        prev, cur = cur, [0] * (n + 1)
    
    return prev[n]



def lcs_length_full_table(X: List[int], Y: List[int]) -> np.ndarray:
    """
    Compute the full LCS DP table (for backtracking / analysis).
    Returns:
        2-D numpy array L of shape (m+1, n+1).
    """
    m, n = len(X), len(Y)
    L = np.zeros((m+1, n+1), dtype=int)

    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i][j-1], L[i-1][j])
    
    return L



def recover_lcs(X: List[int], Y: List[int], L: np.ndarray) -> List[int]:
    """Backtrack through the DP table to recover one actual LCS.

    Args:
        X, Y: The original sequences.
        L: The full DP table from lcs_length_full_table().

    Returns:
        A list of item IDs forming the LCS.
    """
    i, j = len(X), len(Y)
    result = []
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            result.append(X[i - 1])
            i -= 1
            j -= 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return result[::-1]


### Similarity Computation ###

def normalized_lcs_similarity(X: List[int], Y: List[int]) -> float:
    """Compute the normalized LCS similarity between two sequences.

    Similarity = LCS_length(X, Y) / min(len(X), len(Y))

    This gives a value in [0, 1] where 1 means one sequence is a
    subsequence of the other.

    Args:
        X, Y: Sequences of item IDs.

    Returns:
        Normalized similarity score.
    """
    if not X or not Y:
        return 0.0
    length = lcs_length(X, Y)
    return length / min(len(X), len(Y))


def compute_all_similarities(
    target_seq: List[int],
    all_sequences: Dict[int, List[int]],
    exclude_uid: int) -> List[Tuple[int, float, int]]:
    """Compute similarity between a target user and all other users.

    Args:
        target_seq: The target user's training sequence.
        all_sequences: Dictionary of userId -> training sequence.
        exclude_uid: The target user's ID (to skip self-comparison).

    Returns:
        List of (userId, similarity_score, lcs_length), sorted by
        similarity descending.
    """
    results = []
    for uid, seq in all_sequences.items():
        if uid == exclude_uid:
            continue
        l = lcs_length(target_seq, seq)
        sim = l / min(len(target_seq), len(seq)) if min(len(target_seq), len(seq)) > 0 else 0.0
        results.append((uid, sim, l))

    results.sort(key=lambda x: (-x[1], -x[2]))
    return results


def find_top_k_similar(
    target_uid: int,
    target_seq: List[int],
    all_sequences: Dict[int, List[int]],
    k: int = 10) -> List[Tuple[int, float, int]]:
    """Return the top-K most similar users to the target.

    Args:
        target_uid: Target user ID.
        target_seq: Target user's training sequence.
        all_sequences: All users' training sequences.
        k: Number of similar users to return.

    Returns:
        Top-K list of (userId, similarity, lcs_length).
    """
    sims = compute_all_similarities(target_seq, all_sequences, target_uid)
    return sims[:k]


# ─────────────────────────────────────────────
# Batch similarity (for full evaluation)
# ─────────────────────────────────────────────
def compute_similarity_matrix(
    sequences: Dict[int, List[int]],
    user_ids: List[int] = None,
    k: int = 10,
    show_progress: bool = True) -> Dict[int, List[Tuple[int, float, int]]]:
    """Compute top-K similar users for every user in user_ids.

    Args:
        sequences: All training sequences.
        user_ids: Users to compute neighbours for (default: all).
        k: Number of neighbours.
        show_progress: Show a tqdm progress bar.

    Returns:
        Dictionary mapping userId -> top-K list of (neighbour_id, sim, lcs_len).
    """
    if user_ids is None:
        user_ids = list(sequences.keys())

    neighbours: Dict[int, List[Tuple[int, float, int]]] = {}
    it = tqdm(user_ids, desc="Computing similarities") if show_progress else user_ids

    for uid in it:
        neighbours[uid] = find_top_k_similar(uid, sequences[uid], sequences, k)

    return neighbours


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Example from CLRS
    X = [1, 0, 0, 1, 0, 1, 0, 1]
    Y = [0, 1, 0, 1, 1, 0, 1, 1, 0]

    print("X:", X)
    print("Y:", Y)
    print("LCS length:", lcs_length(X, Y))

    L = lcs_length_full_table(X, Y)
    print("LCS:", recover_lcs(X, Y, L))
    print("Similarity:", normalized_lcs_similarity(X, Y))

    # Small user-sequence test
    seqs = {
        1: [10, 20, 30, 40, 50],
        2: [10, 30, 50, 60],
        3: [20, 40, 60, 80],
    }
    print("\n--- User similarity test ---")
    for uid in seqs:
        top = find_top_k_similar(uid, seqs[uid], seqs, k=2)
        print(f"User {uid} -> {top}")