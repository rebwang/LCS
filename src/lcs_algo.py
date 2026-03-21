from typing import List, Dict, Tuple
import numpy as np

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



def recover_lcs():
    pass

def normalized_lcs_similarity():
    pass

def compute_all_similarities():
    pass

def find_top_k_similar():
    pass

def compute_similarity_matrix():
    pass
