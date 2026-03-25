import os
import pandas as pd
from typing import Dict, List, Tuple


DATA_DIR = os.path.join("ml-latest-small")
RATING_THRESHOLD = 3.5      # minimum rating to count as a "positive" interaction
MIN_SEQUENCE_LENGTH = 5     # users with fewer items are excluded


def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load and return the ratings DataFrame.

    Columns: userId, movieId, rating, timestamp
    """
    path = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ratings.csv not found at {path}. "
            f"Please update DATA_DIR to point to your ml-latest-small folder."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} ratings from {df['userId'].nunique()} users")
    return df


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load movie metadata. Columns: movieId, title, genres"""
    path = os.path.join(data_dir, "movies.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"movies.csv not found at {path}. "
            f"Please update DATA_DIR to point to your ml-latest-small folder."
        )
    return pd.read_csv(path)


def build_sequences(
    ratings: pd.DataFrame,
    rating_threshold: float = RATING_THRESHOLD,
    min_length: int = MIN_SEQUENCE_LENGTH,
) -> Dict[int, List[int]]:
    """Convert ratings into per-user ordered viewing sequences.

    Steps:
      1. Keep only ratings >= rating_threshold (positive interactions).
      2. Sort each user's ratings by timestamp.
      3. Extract the ordered list of movieIds.
      4. Filter out users with fewer than min_length items.

    Args:
        ratings: Raw ratings DataFrame.
        rating_threshold: Minimum star rating to include.
        min_length: Minimum sequence length to keep a user.

    Returns:
        Dictionary mapping userId -> list of movieIds in chronological order.
    """
    # Filter to positive interactions
    pos = ratings[ratings["rating"] >= rating_threshold].copy()
    pos.sort_values(["userId", "timestamp"], inplace=True)

    sequences: Dict[int, List[int]] = {}
    for uid, group in pos.groupby("userId"):
        seq = group["movieId"].tolist()
        if len(seq) >= min_length:
            sequences[int(uid)] = seq

    print(f"[INFO] Built sequences for {len(sequences)} users "
          f"(threshold={rating_threshold}, min_length={min_length})")
    return sequences


def train_test_split(
    sequences: Dict[int, List[int]],
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """Hold out the last item of each user's sequence for testing.

    Returns:
        train_seqs: userId -> sequence without the last item
        test_items: userId -> the held-out movieId (ground truth)
    """
    train_seqs: Dict[int, List[int]] = {}
    test_items: Dict[int, int] = {}

    for uid, seq in sequences.items():
        train_seqs[uid] = seq[:-1]
        test_items[uid] = seq[-1]

    print(f"[INFO] Train/test split: {len(train_seqs)} users, "
          f"1 held-out item each")
    return train_seqs, test_items


def get_movie_title_map(movies: pd.DataFrame) -> Dict[int, str]:
    """Return a movieId -> title lookup dictionary."""
    return dict(zip(movies["movieId"], movies["title"]))


if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()
    sequences = build_sequences(ratings)
    train, test = train_test_split(sequences)

    # Print sample
    sample_uid = list(train.keys())[0]
    title_map = get_movie_title_map(movies)
    print(f"\n--- Sample: User {sample_uid} ---")
    print(f"Train sequence length: {len(train[sample_uid])}")
    print(f"First 5 movies: {[title_map.get(m, m) for m in train[sample_uid][:5]]}")
    print(f"Held-out test movie:  {title_map.get(test[sample_uid], test[sample_uid])}")