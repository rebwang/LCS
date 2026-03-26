# visualize_results.py
# Responsible: Yue Liang
# All charts and figures for the final report and presentation.

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Consistent colour palette
LCS_COLOR    = '#2196F3'   # blue  – LCS model
RANDOM_COLOR = '#FF5722'   # orange – random baseline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Figure 1 – Hit Rate@K bar chart (LCS vs Random)
# ---------------------------------------------------------------------------

def plot_hit_rate_comparison(
    lcs_hit_rates: Dict[int, float],
    random_hit_rates: Dict[int, float],
    k_values: List[int],
    save_path: Optional[str] = None,
) -> None:
    """Grouped bar chart comparing LCS and random Hit Rate at each K.

    Args:
        lcs_hit_rates:    {k: hit_rate} for the LCS recommender.
        random_hit_rates: {k: hit_rate} for the random baseline.
        k_values:         K values to display on the x-axis.
        save_path:        File path to save the figure (PNG). If None, shows it.
    """
    x = np.arange(len(k_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    lcs_vals    = [lcs_hit_rates.get(k, 0)    for k in k_values]
    random_vals = [random_hit_rates.get(k, 0) for k in k_values]

    bars_lcs    = ax.bar(x - width / 2, lcs_vals,    width, label='LCS',    color=LCS_COLOR,    alpha=0.85)
    bars_random = ax.bar(x + width / 2, random_vals, width, label='Random', color=RANDOM_COLOR, alpha=0.85)

    # Annotate bar tops
    for bar in bars_lcs:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_random:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('K (recommendation list size)', fontsize=12)
    ax.set_ylabel('Hit Rate@K', fontsize=12)
    ax.set_title('LCS vs Random Baseline – Hit Rate@K', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_values])
    ax.legend()
    ax.set_ylim(0, min(1.0, max(lcs_vals + random_vals) * 1.25 + 0.05))
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(_ensure_output_dir(), 'hit_rate_comparison.png'))


# ---------------------------------------------------------------------------
# Figure 2 – Hit Rate@K line curve
# ---------------------------------------------------------------------------

def plot_hit_rate_curve(
    lcs_hit_rates: Dict[int, float],
    random_hit_rates: Dict[int, float],
    k_values: List[int],
    save_path: Optional[str] = None,
) -> None:
    """Line chart showing how Hit Rate changes as K increases.

    Args:
        lcs_hit_rates:    {k: hit_rate} for the LCS recommender.
        random_hit_rates: {k: hit_rate} for the random baseline.
        k_values:         K values (x-axis).
        save_path:        File path to save the figure.
    """
    lcs_vals    = [lcs_hit_rates.get(k, 0)    for k in k_values]
    random_vals = [random_hit_rates.get(k, 0) for k in k_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, lcs_vals,    marker='o', color=LCS_COLOR,    linewidth=2, label='LCS')
    ax.plot(k_values, random_vals, marker='s', color=RANDOM_COLOR, linewidth=2, label='Random', linestyle='--')

    ax.set_xlabel('K (recommendation list size)', fontsize=12)
    ax.set_ylabel('Hit Rate@K', fontsize=12)
    ax.set_title('Hit Rate@K vs K', fontsize=14, fontweight='bold')
    ax.legend()
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(_ensure_output_dir(), 'hit_rate_curve.png'))


# ---------------------------------------------------------------------------
# Figure 3 – Similarity heatmap (subset of users)
# ---------------------------------------------------------------------------

def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    user_ids: List[int],
    sample_size: int = 30,
    save_path: Optional[str] = None,
) -> None:
    """Heatmap of the user-user LCS similarity matrix.

    Shows a random sample of users so the chart stays readable.

    Args:
        sim_matrix:  Full (n x n) similarity matrix.
        user_ids:    Ordered list of user IDs.
        sample_size: Number of users to display.
        save_path:   File path to save the figure.
    """
    n = len(user_ids)
    sample_size = min(sample_size, n)

    # Take the first `sample_size` users (most active, since main.py sorts by activity)
    idx = list(range(sample_size))
    sub_matrix = sim_matrix[np.ix_(idx, idx)]
    sub_labels  = [str(user_ids[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sub_matrix,
        xticklabels=sub_labels,
        yticklabels=sub_labels,
        cmap='Blues',
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor='white',
        annot=(sample_size <= 15),   # only annotate if few cells
        fmt='.2f',
        ax=ax,
    )
    ax.set_title(
        f'User–User LCS Similarity Matrix (top {sample_size} most active users)',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlabel('User ID', fontsize=11)
    ax.set_ylabel('User ID', fontsize=11)
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(_ensure_output_dir(), 'similarity_heatmap.png'))


# ---------------------------------------------------------------------------
# Figure 4 – Similarity score distribution
# ---------------------------------------------------------------------------

def plot_similarity_distribution(
    sim_matrix: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Histogram of all pairwise (off-diagonal) similarity scores.

    Args:
        sim_matrix: Full (n x n) similarity matrix.
        save_path:  File path to save the figure.
    """
    n = sim_matrix.shape[0]
    # Extract upper-triangle (excluding diagonal)
    upper_idx = np.triu_indices(n, k=1)
    scores = sim_matrix[upper_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=40, color=LCS_COLOR, alpha=0.8, edgecolor='white')
    ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {scores.mean():.3f}')
    ax.set_xlabel('Normalised LCS Similarity', fontsize=12)
    ax.set_ylabel('Number of User Pairs', fontsize=12)
    ax.set_title('Distribution of Pairwise LCS Similarity Scores', fontsize=14, fontweight='bold')
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(_ensure_output_dir(), 'similarity_distribution.png'))


# ---------------------------------------------------------------------------
# Figure 5 – Sequence comparison (two users + their LCS)
# ---------------------------------------------------------------------------

def plot_sequence_comparison(
    seq_a: List[int],
    seq_b: List[int],
    lcs_seq: List[int],
    user_a_id: int,
    user_b_id: int,
    movie_titles: Optional[Dict[int, str]] = None,
    max_items: int = 15,
    save_path: Optional[str] = None,
) -> None:
    """Visualise two user sequences and their common subsequence side-by-side.

    Each sequence is drawn as a horizontal row of labelled boxes.  Items that
    are part of the LCS are highlighted in gold.

    Args:
        seq_a, seq_b:  Viewing sequences for the two users.
        lcs_seq:       The LCS of seq_a and seq_b.
        user_a_id:     User ID for seq_a (used in the row label).
        user_b_id:     User ID for seq_b (used in the row label).
        movie_titles:  Optional {movieId: title} dict for readable labels.
        max_items:     Truncate sequences to this length for readability.
        save_path:     File path to save the figure.
    """
    seq_a   = seq_a[:max_items]
    seq_b   = seq_b[:max_items]
    lcs_set = set(lcs_seq)

    def label(movie_id: int) -> str:
        if movie_titles and movie_id in movie_titles:
            title = movie_titles[movie_id]
            # Shorten to two words
            words = title.split()
            return ' '.join(words[:2]) + ('…' if len(words) > 2 else '')
        return str(movie_id)

    def draw_row(ax, sequence, y, row_label, highlight_set):
        ax.text(-0.5, y, row_label, ha='right', va='center', fontsize=9, fontweight='bold')
        for x, movie_id in enumerate(sequence):
            color = '#FFD700' if movie_id in highlight_set else '#BBDEFB'
            rect = mpatches.FancyBboxPatch(
                (x - 0.45, y - 0.4), 0.9, 0.8,
                boxstyle='round,pad=0.05',
                facecolor=color, edgecolor='#555', linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(x, y, label(movie_id), ha='center', va='center',
                    fontsize=6.5, wrap=True)

    n_cols = max(len(seq_a), len(seq_b))
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.9), 4))
    ax.set_xlim(-1, n_cols)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')

    draw_row(ax, seq_a, 2.0, f'User {user_a_id}', lcs_set)
    draw_row(ax, seq_b, 1.0, f'User {user_b_id}', lcs_set)

    # Legend
    lcs_patch  = mpatches.Patch(facecolor='#FFD700', edgecolor='#555', label='In LCS (shared)')
    rest_patch = mpatches.Patch(facecolor='#BBDEFB', edgecolor='#555', label='Unique to user')
    ax.legend(handles=[lcs_patch, rest_patch], loc='lower right', fontsize=9)

    ax.set_title(
        f'Sequence Comparison: User {user_a_id} vs User {user_b_id}  '
        f'(LCS length = {len(lcs_seq)})',
        fontsize=12, fontweight='bold', pad=12,
    )

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(_ensure_output_dir(), 'sequence_comparison.png'))


# ---------------------------------------------------------------------------
# Figure 6 – Precision & Recall summary table as a chart
# ---------------------------------------------------------------------------

def plot_metrics_table(
    lcs_metrics: Dict[str, Dict[int, float]],
    random_metrics: Dict[str, Dict[int, float]],
    k_values: List[int],
    save_path: Optional[str] = None,
) -> None:
    """Render a colour-coded table of Precision and Recall at each K.

    Args:
        lcs_metrics:    Output of recommendation.evaluate_all_metrics() for LCS.
        random_metrics: Output of recommendation.evaluate_all_metrics() for random.
        k_values:       K values shown as columns.
        save_path:      File path to save the figure.
    """
    metric_names = ['hit_rate', 'precision', 'recall']
    display_names = ['Hit Rate', 'Precision', 'Recall']

    rows = []
    row_labels = []
    for metric, display in zip(metric_names, display_names):
        lcs_row    = [f'{lcs_metrics[metric].get(k, 0):.4f}'    for k in k_values]
        random_row = [f'{random_metrics[metric].get(k, 0):.4f}' for k in k_values]
        rows.append(lcs_row)
        rows.append(random_row)
        row_labels.append(f'LCS – {display}')
        row_labels.append(f'Random – {display}')

    col_labels = [f'K={k}' for k in k_values]

    fig, ax = plt.subplots(figsize=(max(7, len(k_values) * 1.4), len(rows) * 0.55 + 1.5))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Colour LCS rows blue, random rows orange
    for (row, col), cell in table.get_celld().items():
        if col == -1:  # row label column
            if 'LCS' in row_labels[row - 1] if row > 0 else False:
                cell.set_facecolor('#DCEEFB')
            else:
                cell.set_facecolor('#FFF3E0')
        elif row == 0:
            cell.set_facecolor('#ECEFF1')
            cell.set_text_props(fontweight='bold')

    ax.set_title('Evaluation Metrics Summary', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(_ensure_output_dir(), 'metrics_table.png'))


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: str) -> None:
    """Save figure to disk and close it."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {save_path}')
