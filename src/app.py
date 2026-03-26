# app.py
# Streamlit frontend for the LCS-based Recommendation System.
# Run with:  streamlit run src/app.py

import io
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from data_processing import (
    load_ratings, load_movies,
    build_user_sequences, train_test_split, get_top_users_by_activity,
)
from lcs_algo import (
    compute_similarity_matrix, recover_lcs,
    lcs_length_full_table,
)
from recommendation import (
    recommend_all_users, random_recommend, evaluate_all_metrics,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml-latest-small')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='LCS Recommender',
    page_icon='🎬',
    layout='wide',
)

st.title('🎬 LCS-Based Movie Recommendation System')
st.caption('CS 5800 Final Project — Group 10  |  Soonbee Hwang · Rebecca Wang · Yue Liang')

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Parameters')
    max_users   = st.slider('Max users',              20,  300, 150, step=10)
    max_seq_len = st.slider('Max sequence length',    10,  100,  50, step=5)
    min_seq_len = st.slider('Min sequence length',     5,   30,  10, step=5)
    top_k_users = st.slider('Neighbour size (K users)', 3,  30,  10)
    top_n_items = st.slider('Recommendation list size', 5,  50,  20, step=5)
    k_values    = st.multiselect('K values for evaluation',
                                 [1, 5, 10, 20, 50], default=[1, 5, 10, 20])
    run = st.button('▶ Run Pipeline', type='primary', use_container_width=True)

# ── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(max_users, max_seq_len, min_seq_len,
                 top_k_users, top_n_items, k_values_tuple):
    k_vals = list(k_values_tuple)
    ratings_df   = load_ratings(os.path.join(DATA_DIR, 'ratings.csv'))
    movies_df    = load_movies(os.path.join(DATA_DIR,  'movies.csv'))
    movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))

    sequences = build_user_sequences(ratings_df, min_len=min_seq_len, max_len=max_seq_len)
    sequences = get_top_users_by_activity(sequences, max_users)
    train_seqs, test_labels = train_test_split(sequences)
    all_items = list({mid for seq in train_seqs.values() for mid in seq})

    sim_matrix, user_ids = compute_similarity_matrix(train_seqs)
    lcs_recs    = recommend_all_users(user_ids, sim_matrix, train_seqs,
                                      top_k_users=top_k_users, top_n_items=top_n_items)
    random_recs = random_recommend(train_seqs, all_items, top_n_items=top_n_items)
    lcs_metrics    = evaluate_all_metrics(lcs_recs,    test_labels, k_vals)
    random_metrics = evaluate_all_metrics(random_recs, test_labels, k_vals)

    return dict(train_seqs=train_seqs, test_labels=test_labels,
                user_ids=user_ids, sim_matrix=sim_matrix,
                lcs_recs=lcs_recs, lcs_metrics=lcs_metrics,
                random_metrics=random_metrics, movie_titles=movie_titles,
                k_values=k_vals)


# ── State ─────────────────────────────────────────────────────────────────────
if 'data' not in st.session_state:
    st.session_state['data'] = None

if run:
    if not k_values:
        st.sidebar.error('Select at least one K value.')
    else:
        with st.spinner('Running pipeline … (first run ~10 s)'):
            st.session_state['data'] = run_pipeline(
                max_users, max_seq_len, min_seq_len,
                top_k_users, top_n_items, tuple(sorted(k_values)),
            )

data = st.session_state['data']

if data is None:
    st.info('👈 Adjust parameters in the sidebar and click **Run Pipeline** to start.')
    st.stop()

# unpack common refs
user_ids     = data['user_ids']
train_seqs   = data['train_seqs']
sim_matrix   = data['sim_matrix']
movie_titles = data['movie_titles']

def short_title(mid, n=2):
    words = movie_titles.get(mid, str(mid)).split()
    return ' '.join(words[:n]) + ('…' if len(words) > n else '')

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '🎯 Recommendations',
    '🔗 Similarity Analysis',
    '📊 Evaluation',
    '📐 Algorithm',
])

# ═════════════════════════════════════════════════════════════════════════════
# Tab 1 – Recommendations + CSV export
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader('User Recommendations')

    lcs_recs    = data['lcs_recs']
    test_labels = data['test_labels']

    selected_uid = st.selectbox('Select a user', sorted(user_ids), key='t1_uid')

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('**Viewing history (training sequence)**')
        history = train_seqs.get(selected_uid, [])
        st.dataframe(
            [{'#': i+1, 'Movie ID': mid, 'Title': movie_titles.get(mid, '—')}
             for i, mid in enumerate(history)],
            use_container_width=True, hide_index=True,
        )

    with col_b:
        st.markdown('**Top recommendations (LCS)**')
        recs         = lcs_recs.get(selected_uid, [])
        ground_truth = test_labels.get(selected_uid)
        rec_rows = [
            {'Rank': rank, 'Movie ID': mid,
             'Title': movie_titles.get(mid, '—'),
             'Hit': '✅' if mid == ground_truth else ''}
            for rank, mid in enumerate(recs, 1)
        ]
        st.dataframe(rec_rows, use_container_width=True, hide_index=True)

        if ground_truth:
            gt_title = movie_titles.get(ground_truth, str(ground_truth))
            if ground_truth in recs:
                st.success(f'Ground truth **"{gt_title}"** found in recommendations ✅')
            else:
                st.warning(f'Ground truth **"{gt_title}"** was NOT recommended')

    # ── CSV export ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('**Export all recommendations as CSV**')

    @st.cache_data(show_spinner=False)
    def build_csv(lcs_recs_frozen, test_labels_frozen, movie_titles_frozen):
        rows = []
        for uid, recs in lcs_recs_frozen:
            gt = test_labels_frozen.get(uid)
            for rank, mid in enumerate(recs, 1):
                rows.append({
                    'userId':        uid,
                    'rank':          rank,
                    'movieId':       mid,
                    'title':         movie_titles_frozen.get(mid, ''),
                    'is_ground_truth': int(mid == gt) if gt else 0,
                })
        return pd.DataFrame(rows).to_csv(index=False).encode('utf-8')

    csv_bytes = build_csv(
        tuple(sorted(lcs_recs.items())),
        test_labels,
        movie_titles,
    )

    col_dl1, col_dl2, _ = st.columns([1, 1, 3])
    col_dl1.download_button(
        label='⬇ Download recommendations.csv',
        data=csv_bytes,
        file_name='lcs_recommendations.csv',
        mime='text/csv',
    )

    # also export evaluation summary
    k_vals_used    = data['k_values']
    lcs_metrics    = data['lcs_metrics']
    random_metrics = data['random_metrics']

    eval_rows = []
    for metric in ['hit_rate', 'precision', 'recall']:
        for model, mdata in [('LCS', lcs_metrics), ('Random', random_metrics)]:
            row = {'model': model, 'metric': metric}
            for k in k_vals_used:
                row[f'at_k{k}'] = round(mdata[metric].get(k, 0), 6)
            eval_rows.append(row)
    eval_csv = pd.DataFrame(eval_rows).to_csv(index=False).encode('utf-8')

    col_dl2.download_button(
        label='⬇ Download evaluation.csv',
        data=eval_csv,
        file_name='lcs_evaluation.csv',
        mime='text/csv',
    )

# ═════════════════════════════════════════════════════════════════════════════
# Tab 2 – Similarity Analysis
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader('User–User Similarity')

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('**Similarity heatmap** (top 30 most active users)')
        n_show = min(30, len(user_ids))
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(sim_matrix[:n_show, :n_show],
                    xticklabels=[str(u) for u in user_ids[:n_show]],
                    yticklabels=[str(u) for u in user_ids[:n_show]],
                    cmap='Blues', vmin=0, vmax=1,
                    linewidths=0.3, linecolor='white', ax=ax)
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', rotation=0,  labelsize=6)
        ax.set_title('LCS Similarity Matrix', fontsize=11)
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('**Similarity distribution**')
        upper  = np.triu_indices(len(user_ids), k=1)
        scores = sim_matrix[upper]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.hist(scores, bins=35, color='#2196F3', alpha=0.85, edgecolor='white')
        ax2.axvline(scores.mean(), color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean = {scores.mean():.3f}')
        ax2.set_xlabel('Similarity'); ax2.set_ylabel('# Pairs')
        ax2.legend(fontsize=9)
        ax2.set_title('Pairwise Similarity Distribution', fontsize=11)
        st.pyplot(fig2); plt.close(fig2)

        st.metric('Mean similarity',       f'{scores.mean():.4f}')
        st.metric('Max similarity',        f'{scores.max():.4f}')
        st.metric('Pairs with sim > 0.2',  f'{(scores > 0.2).sum()} / {len(scores)}')

    st.divider()
    st.markdown('**Sequence comparison between two users**')
    c1, c2 = st.columns(2)
    uid_a = c1.selectbox('User A', sorted(user_ids), key='ua')
    uid_b = c2.selectbox('User B', sorted(user_ids), index=min(1, len(user_ids)-1), key='ub')

    if uid_a != uid_b:
        seq_a   = train_seqs.get(uid_a, [])
        seq_b   = train_seqs.get(uid_b, [])
        lcs_seq = recover_lcs(seq_a, seq_b)
        sim_val = sim_matrix[user_ids.index(uid_a)][user_ids.index(uid_b)]

        st.caption(f'LCS length: **{len(lcs_seq)}**  |  Similarity: **{sim_val:.4f}**')

        sa, sb  = seq_a[:12], seq_b[:12]
        lcs_set = set(lcs_seq)
        n_cols  = max(len(sa), len(sb))
        figc, axc = plt.subplots(figsize=(max(8, n_cols * 0.85), 3.5))
        axc.set_xlim(-1, n_cols); axc.set_ylim(0, 3); axc.axis('off')
        for row_y, seq, lbl in [(2.2, sa, f'User {uid_a}'), (1.0, sb, f'User {uid_b}')]:
            axc.text(-0.6, row_y, lbl, ha='right', va='center',
                     fontsize=9, fontweight='bold')
            for x, mid in enumerate(seq):
                color = '#FFD700' if mid in lcs_set else '#BBDEFB'
                axc.add_patch(mpatches.FancyBboxPatch(
                    (x - 0.44, row_y - 0.38), 0.88, 0.76,
                    boxstyle='round,pad=0.05', facecolor=color,
                    edgecolor='#666', linewidth=0.7))
                axc.text(x, row_y, short_title(mid),
                         ha='center', va='center', fontsize=6.2)
        axc.legend(handles=[
            mpatches.Patch(facecolor='#FFD700', edgecolor='#666', label='In LCS'),
            mpatches.Patch(facecolor='#BBDEFB', edgecolor='#666', label='Unique'),
        ], loc='lower right', fontsize=8)
        st.pyplot(figc); plt.close(figc)
    else:
        st.info('Select two different users to compare.')

# ═════════════════════════════════════════════════════════════════════════════
# Tab 3 – Evaluation
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader('Evaluation Metrics')

    lcs_metrics    = data['lcs_metrics']
    random_metrics = data['random_metrics']
    k_vals_used    = data['k_values']
    best_k         = max(k_vals_used)

    cols = st.columns(3)
    for col, metric, label in zip(cols,
                                   ['hit_rate', 'precision', 'recall'],
                                   ['Hit Rate', 'Precision', 'Recall']):
        lcs_val  = lcs_metrics[metric].get(best_k, 0)
        rand_val = random_metrics[metric].get(best_k, 0)
        col.metric(f'{label}@{best_k} (LCS)', f'{lcs_val:.4f}',
                   delta=f'{lcs_val - rand_val:+.4f} vs random')

    st.divider()
    cc1, cc2 = st.columns(2)
    lcs_vals  = [lcs_metrics['hit_rate'].get(k, 0)    for k in k_vals_used]
    rand_vals = [random_metrics['hit_rate'].get(k, 0) for k in k_vals_used]

    with cc1:
        st.markdown('**Hit Rate@K — bar chart**')
        x, w = np.arange(len(k_vals_used)), 0.35
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        b1 = ax3.bar(x - w/2, lcs_vals,  w, label='LCS',    color='#2196F3', alpha=0.85)
        b2 = ax3.bar(x + w/2, rand_vals, w, label='Random', color='#FF5722', alpha=0.85)
        for b in list(b1) + list(b2):
            ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                     f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'K={k}' for k in k_vals_used])
        ax3.set_ylabel('Hit Rate@K'); ax3.legend()
        ax3.yaxis.grid(True, linestyle='--', alpha=0.4); ax3.set_axisbelow(True)
        st.pyplot(fig3); plt.close(fig3)

    with cc2:
        st.markdown('**Hit Rate@K — line chart**')
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(k_vals_used, lcs_vals,  marker='o', color='#2196F3', linewidth=2, label='LCS')
        ax4.plot(k_vals_used, rand_vals, marker='s', color='#FF5722', linewidth=2,
                 linestyle='--', label='Random')
        ax4.set_xlabel('K'); ax4.set_ylabel('Hit Rate@K'); ax4.legend()
        ax4.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax4.yaxis.grid(True, linestyle='--', alpha=0.4); ax4.set_axisbelow(True)
        st.pyplot(fig4); plt.close(fig4)

    st.divider()
    st.markdown('**Full metrics table**')
    eval_rows = []
    for metric in ['hit_rate', 'precision', 'recall']:
        for model, mdata in [('LCS', lcs_metrics), ('Random', random_metrics)]:
            row = {'Model': model, 'Metric': metric.replace('_', ' ').title()}
            for k in k_vals_used:
                row[f'@{k}'] = f"{mdata[metric].get(k, 0):.4f}"
            eval_rows.append(row)
    st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 4 – Algorithm explanation + DP table visualiser
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader('Algorithm: Longest Common Subsequence (LCS)')

    # ── 1. Theory ─────────────────────────────────────────────────────────────
    with st.expander('📖 What is LCS?', expanded=True):
        st.markdown(r'''
**Definition**
Given two sequences $X = \langle x_1, x_2, \ldots, x_m \rangle$ and
$Y = \langle y_1, y_2, \ldots, y_n \rangle$, the **Longest Common Subsequence**
is the longest sequence $Z$ such that $Z$ is a subsequence of both $X$ and $Y$.
Elements do **not** need to be contiguous — only their relative order matters.

**Dynamic Programming Recurrence** (CLRS §15.4)

$$
L[i][j] = \begin{cases}
0 & \text{if } i = 0 \text{ or } j = 0 \\
L[i-1][j-1] + 1 & \text{if } x_i = y_j \\
\max\bigl(L[i-1][j],\; L[i][j-1]\bigr) & \text{otherwise}
\end{cases}
$$

**Time complexity:** $O(mn)$ · **Space complexity:** $O(mn)$ (full table) or $O(\min(m,n))$ (optimised)

**How we use it for recommendations**
Each user's viewing history is an ordered sequence of movie IDs.
LCS length is used as a similarity measure: the more movies two users watched in the same order, the more similar they are.

$$\text{similarity}(U, V) = \frac{\text{LCS}(S_U,\, S_V)}{\max(|S_U|, |S_V|)} \in [0, 1]$$

For a target user, we find the most similar neighbours and recommend items they watched that the target user hasn't seen yet.
''')

    st.divider()

    # ── 2. Interactive DP table ───────────────────────────────────────────────
    st.markdown('### Interactive DP Table')
    st.caption('Pick two users from the dataset and visualise the LCS DP table for a prefix of their sequences.')

    dp_c1, dp_c2, dp_c3 = st.columns([2, 2, 1])
    dp_uid_a  = dp_c1.selectbox('User A', sorted(user_ids), key='dp_ua')
    dp_uid_b  = dp_c2.selectbox('User B', sorted(user_ids),
                                 index=min(1, len(user_ids)-1), key='dp_ub')
    prefix_n  = dp_c3.number_input('Prefix length', min_value=3, max_value=15,
                                    value=8, step=1)

    if dp_uid_a == dp_uid_b:
        st.info('Select two different users.')
        st.stop()

    seq_a = train_seqs.get(dp_uid_a, [])[:prefix_n]
    seq_b = train_seqs.get(dp_uid_b, [])[:prefix_n]

    if not seq_a or not seq_b:
        st.warning('One of the sequences is empty.')
        st.stop()

    L       = lcs_length_full_table(seq_a, seq_b)
    lcs_seq = recover_lcs(seq_a, seq_b, L)

    # Build backtracking path
    path_cells = set()
    i, j = len(seq_a), len(seq_b)
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            path_cells.add((i, j))
            i -= 1; j -= 1
        elif L[i-1][j] >= L[i][j-1]:
            path_cells.add((i, j))
            i -= 1
        else:
            path_cells.add((i, j))
            j -= 1

    # ── draw DP table ─────────────────────────────────────────────────────────
    m, n   = len(seq_a), len(seq_b)
    cell_w = 0.9
    fig_w  = (n + 2) * cell_w + 1
    fig_h  = (m + 2) * cell_w + 0.5
    fig5, ax5 = plt.subplots(figsize=(fig_w, fig_h))
    ax5.set_xlim(-1.5, n + 0.5)
    ax5.set_ylim(-0.5, m + 1.5)
    ax5.invert_yaxis()
    ax5.axis('off')
    ax5.set_title(
        f'LCS DP Table  —  User {dp_uid_a} (rows) vs User {dp_uid_b} (cols)\n'
        f'LCS length = {int(L[m][n])}  |  Similarity = {int(L[m][n]) / max(m, n):.3f}',
        fontsize=10, fontweight='bold', pad=8,
    )

    MATCH_COLOR  = '#4CAF50'  # green  – matched diagonal
    PATH_COLOR   = '#FFF9C4'  # yellow – backtracking path
    HEADER_COLOR = '#E3F2FD'  # light blue – row/col headers
    CELL_COLOR   = '#FFFFFF'  # white  – regular cells

    # column headers (sequence B)
    ax5.text(-1.0, 0.5, '', ha='center', va='center', fontsize=7)   # top-left corner
    ax5.text(-0.5, 0.5, 'ε', ha='center', va='center', fontsize=9, color='#666')
    for jj, mid in enumerate(seq_b):
        ax5.add_patch(mpatches.FancyBboxPatch(
            (jj + 0.5, 0.1), 0.85, 0.8,
            boxstyle='round,pad=0.02', facecolor=HEADER_COLOR, edgecolor='#BBB', linewidth=0.6))
        ax5.text(jj + 1, 0.5, short_title(mid, n=1), ha='center', va='center',
                 fontsize=6.5, color='#1A237E')
        ax5.text(jj + 1, -0.1, str(mid), ha='center', va='center',
                 fontsize=5, color='#888')

    for ii in range(m + 1):
        # row header (sequence A item or ε)
        if ii == 0:
            label = 'ε'
        else:
            label = short_title(seq_a[ii-1], n=1)
            ax5.text(-1.0, ii + 0.5, str(seq_a[ii-1]), ha='center', va='center',
                     fontsize=5, color='#888')
        ax5.add_patch(mpatches.FancyBboxPatch(
            (-1.4, ii + 0.1), 0.85, 0.8,
            boxstyle='round,pad=0.02', facecolor=HEADER_COLOR, edgecolor='#BBB', linewidth=0.6))
        ax5.text(-1.0, ii + 0.5, label, ha='center', va='center',
                 fontsize=6.5, color='#1A237E')

        for jj in range(n + 1):
            val = int(L[ii][jj])
            is_match = (ii > 0 and jj > 0 and seq_a[ii-1] == seq_b[jj-1])
            is_path  = (ii, jj) in path_cells

            if is_match:
                fc = MATCH_COLOR
                tc = 'white'
                fw = 'bold'
            elif is_path:
                fc = PATH_COLOR
                tc = '#333'
                fw = 'normal'
            else:
                fc = CELL_COLOR
                tc = '#333'
                fw = 'normal'

            ax5.add_patch(mpatches.FancyBboxPatch(
                (jj - 0.4, ii + 0.1), 0.8, 0.8,
                boxstyle='round,pad=0.02', facecolor=fc,
                edgecolor='#CCC', linewidth=0.6))
            ax5.text(jj - 0.0, ii + 0.5, str(val),
                     ha='center', va='center', fontsize=9,
                     color=tc, fontweight=fw)

    # legend
    ax5.legend(handles=[
        mpatches.Patch(facecolor=MATCH_COLOR,  edgecolor='#AAA', label='Match (diagonal +1)'),
        mpatches.Patch(facecolor=PATH_COLOR,   edgecolor='#AAA', label='Backtracking path'),
        mpatches.Patch(facecolor=HEADER_COLOR, edgecolor='#AAA', label='Sequence items'),
    ], loc='lower right', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    st.pyplot(fig5); plt.close(fig5)

    # ── recovered LCS ─────────────────────────────────────────────────────────
    st.markdown(f'**Recovered LCS** ({len(lcs_seq)} items):')
    if lcs_seq:
        lcs_df = pd.DataFrame([
            {'Position': i+1, 'Movie ID': mid, 'Title': movie_titles.get(mid, '—')}
            for i, mid in enumerate(lcs_seq)
        ])
        st.dataframe(lcs_df, use_container_width=True, hide_index=True)
    else:
        st.info('No common items in these two prefixes.')

    st.divider()

    # ── 3. Algorithm walkthrough ──────────────────────────────────────────────
    with st.expander('🔍 Step-by-step walkthrough'):
        st.markdown(r'''
**How to read the DP table above:**

1. **Row 0 / Column 0** are all 0 — base case (empty sequence has LCS length 0 with anything).
2. **Green cell** at position $(i, j)$: $x_i = y_j$, so
   $L[i][j] = L[i-1][j-1] + 1$ (diagonal + 1).
3. **White cell**: $x_i \neq y_j$, so $L[i][j] = \max(L[i-1][j],\; L[i][j-1])$
   (take the better of "skip from above" or "skip from left").
4. **Yellow cells** trace the backtracking path from $L[m][n]$ back to $L[0][0]$,
   recovering the actual LCS sequence.

**Recommendation logic:**
- Compute this table for every pair of users (using their full training sequences).
- Normalise: $\text{sim}(U,V) = L[m][n] / \max(m, n)$.
- For target user $U$, pick the top-$K$ most similar users.
- Collect movies those neighbours watched that $U$ hasn't seen, weighted by similarity score.
- Return the top-$N$ highest-scored movies.
''')
