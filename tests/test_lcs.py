import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lcs_algo import (
    lcs_length, lcs_length_full_table,
    recover_lcs, normalized_lcs_similarity,
    compute_similarity_matrix, find_top_k_similar,
)


def test_lcs_length_basic():
    assert lcs_length([1, 2, 3, 4], [1, 3, 4]) == 3

def test_lcs_length_empty():
    assert lcs_length([], [1, 2, 3]) == 0
    assert lcs_length([1, 2], []) == 0

def test_lcs_length_identical():
    seq = [10, 20, 30]
    assert lcs_length(seq, seq) == 3

def test_lcs_length_no_common():
    assert lcs_length([1, 2], [3, 4]) == 0

def test_full_table_matches_length():
    X, Y = [1, 2, 3, 4, 5], [2, 4, 5]
    L = lcs_length_full_table(X, Y)
    assert L[len(X)][len(Y)] == lcs_length(X, Y)

def test_recover_lcs():
    X, Y = [1, 3, 4, 5, 7], [1, 4, 5, 7]
    lcs = recover_lcs(X, Y)
    assert lcs == [1, 4, 5, 7]

def test_normalized_similarity_range():
    sim = normalized_lcs_similarity([1, 2, 3], [2, 3, 4])
    assert 0.0 <= sim <= 1.0

def test_normalized_similarity_identical():
    seq = [1, 2, 3, 4]
    assert normalized_lcs_similarity(seq, seq) == 1.0

def test_normalized_similarity_empty():
    assert normalized_lcs_similarity([], [1, 2]) == 0.0

def test_similarity_matrix_shape():
    seqs = {1: [1, 2, 3], 2: [2, 3, 4], 3: [1, 3, 5]}
    matrix, user_ids = compute_similarity_matrix(seqs)
    assert matrix.shape == (3, 3)
    assert set(user_ids) == {1, 2, 3}

def test_similarity_matrix_diagonal():
    seqs = {1: [1, 2, 3], 2: [4, 5, 6]}
    matrix, _ = compute_similarity_matrix(seqs)
    assert matrix[0][0] == 1.0
    assert matrix[1][1] == 1.0

def test_similarity_matrix_symmetric():
    seqs = {1: [1, 2, 3], 2: [2, 3, 4], 3: [1, 5, 6]}
    matrix, _ = compute_similarity_matrix(seqs)
    for i in range(3):
        for j in range(3):
            assert abs(matrix[i][j] - matrix[j][i]) < 1e-9

def test_find_top_k_similar():
    seqs = {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [5, 6, 7, 8]}
    matrix, user_ids = compute_similarity_matrix(seqs)
    top = find_top_k_similar(1, user_ids, matrix, k=2)
    assert len(top) == 2
    # User 2 is most similar to user 1 (identical sequence)
    assert top[0][0] == 2
    assert top[0][1] == 1.0
