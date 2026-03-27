# data_processing.py
# Responsible: Soonbee Hwang
# Loads the MovieLens dataset and constructs ordered user viewing sequences.

import pandas as pd
from typing import Dict, List, Tuple


def load_ratings(ratings_path: str) -> pd.DataFrame:
    """Load ratings CSV and sort each user's history by timestamp."""
    df = pd.read_csv(ratings_path)
    df = df.sort_values(['userId', 'timestamp'])
    return df


def load_movies(movies_path: str) -> pd.DataFrame:
    """Load movies CSV (movieId, title, genres)."""
    return pd.read_csv(movies_path)


def build_user_sequences(
    df: pd.DataFrame,
    min_len: int = 10,
    max_len: int = 50,
) -> Dict[int, List[int]]:
    """
    Build an ordered list of movieIds for each user.

    Args:
        df:      Ratings dataframe sorted by (userId, timestamp).
        min_len: Drop users with fewer than this many ratings.
        max_len: Keep only the most recent `max_len` items per user.
                 None means keep all items.

    Returns:
        {userId: [movieId, ...]} sorted chronologically.
    """
    sequences: Dict[int, List[int]] = {}
    for user_id, group in df.groupby('userId'):
        movies = group['movieId'].tolist()
        if len(movies) < min_len:
            continue
        if max_len is not None:
            movies = movies[-max_len:]
        sequences[int(user_id)] = movies
    return sequences


def build_user_interaction_sequences(
    df: pd.DataFrame,
    min_len: int = 10,
    max_len: int = 50,
) -> Dict[int, List[Tuple[int, float]]]:
    """Build an ordered list of (movieId, rating) pairs for each user."""
    interactions: Dict[int, List[Tuple[int, float]]] = {}
    for user_id, group in df.groupby('userId'):
        pairs = list(zip(group['movieId'].tolist(), group['rating'].tolist()))
        if len(pairs) < min_len:
            continue
        if max_len is not None:
            pairs = pairs[-max_len:]
        interactions[int(user_id)] = [(int(movie_id), float(rating)) for movie_id, rating in pairs]
    return interactions


def train_test_split(
    sequences: Dict[int, List[int]]
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Leave-one-out split: the last item in each sequence is the test label;
    the rest is the training sequence.

    Returns:
        train_sequences: {userId: [movieId, ...]} (all but last item)
        test_labels:     {userId: movieId}        (last item)
    """
    train: Dict[int, List[int]] = {}
    test: Dict[int, int] = {}
    for uid, seq in sequences.items():
        if len(seq) < 2:
            continue
        train[uid] = seq[:-1]
        test[uid] = seq[-1]
    return train, test


def train_test_split_interactions(
    interactions: Dict[int, List[Tuple[int, float]]]
) -> Tuple[
    Dict[int, List[int]],
    Dict[int, List[Tuple[int, float]]],
    Dict[int, int],
]:
    """Leave-one-out split for interaction histories.

    Returns:
        train_sequences:    {userId: [movieId, ...]}
        train_interactions: {userId: [(movieId, rating), ...]}
        test_labels:        {userId: movieId}
    """
    train_sequences: Dict[int, List[int]] = {}
    train_interactions: Dict[int, List[Tuple[int, float]]] = {}
    test_labels: Dict[int, int] = {}

    for uid, seq in interactions.items():
        if len(seq) < 2:
            continue
        train_interactions[uid] = seq[:-1]
        train_sequences[uid] = [movie_id for movie_id, _rating in seq[:-1]]
        test_labels[uid] = seq[-1][0]

    return train_sequences, train_interactions, test_labels


def get_top_users_by_activity(
    sequences: Dict[int, List[int]], n: int
) -> Dict[int, List[int]]:
    """Return the n users with the longest sequences (most active)."""
    sorted_users = sorted(sequences.items(), key=lambda x: len(x[1]), reverse=True)
    return dict(sorted_users[:n])
