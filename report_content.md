# CS 5800 Final Project Report
## Sequence-Based Behavioral Recommendation Using Longest Common Subsequence

**Group 10 — Northeastern University**
Soonbee Hwang · Rebecca Wang · Yue Liang

---

## 1. Introduction

### 1.1 Context and Research Question

Recommendation systems are among the most pervasive applications of algorithms in modern technology, powering what billions of users watch, read, and buy every day. Traditional collaborative filtering approaches measure user similarity through co-occurrence counts or matrix factorization, treating each rating as an independent event. These methods discard something potentially valuable: the **order** in which a user consumed content. A user who watched a series of science-fiction films before switching to documentaries has a meaningfully different behavioral pattern from a user who watched those same films in reverse order.

This project investigates whether a classical dynamic programming algorithm from combinatorics — the **Longest Common Subsequence (LCS)** — can serve as a principled measure of sequential behavioral similarity between users, and whether that similarity can drive a useful movie recommendation system.

**Research Question:** *Can sequence similarity measured by the LCS algorithm accurately predict a user's next content choice?*

Our approach models each user's viewing history as an ordered sequence of movie IDs. We then compute pairwise user similarity using normalized LCS scores, identify the most behaviorally similar neighbors, and predict the next item a user is likely to watch based on what those neighbors watched.

### 1.2 Personal Investment

**Rebecca Wang** is personally invested in this question because of the intersection it creates between the theoretical algorithms studied in CS 5800 and real-world system design. Having implemented various DP algorithms in coursework settings, she wanted to explore whether a textbook algorithm like LCS — usually introduced in the abstract context of DNA alignment and text diffing — could meaningfully solve a problem that affects millions of users. Working on the core algorithm module reinforced her understanding of space optimization in dynamic programming, a concept she expects to apply in software engineering roles where memory constraints are a practical concern.

**Soonbee Hwang** was drawn to this topic by the data engineering challenge at its foundation. Before any algorithm can run, user behavior must be represented faithfully — a challenge that proved more nuanced than expected. Deciding which ratings count as "positive feedback," how to handle very long histories, and how to structure the preprocessing pipeline all required careful thought about what the data actually represents. This work made real a lesson that is easy to understate in coursework: data quality and feature engineering often matter more than the choice of algorithm.

**Yue Liang** chose to focus on the recommendation and evaluation components because of a personal interest in how algorithmic systems are measured and held accountable. Designing the weighted scoring function — balancing similarity, recency, rating strength, and popularity — required thinking carefully about what "a good recommendation" actually means. Implementing multiple evaluation metrics (Hit Rate, NDCG, MRR) and comparing against a random baseline made it clear that the standard of success matters as much as the method itself.

This topic is important beyond the classroom because recommendation systems shape what information people encounter. Studying a transparent, interpretable approach (LCS is fully explainable at each step) provides a useful contrast to opaque neural methods and raises questions about when simplicity and interpretability should be valued over raw performance.

---

## 2. Technical Discussion and Analysis

### 2.1 Use of Generative AI

We used generative AI (ChatGPT) in two targeted ways during the project.

**Initial exploration prompt:** We began by asking: *"How is LCS typically used in algorithm courses, and have there been any applications of LCS to recommendation systems?"* The AI's response confirmed that LCS is standardly taught as a string/sequence matching algorithm (DNA alignment, diff tools, version control), and noted that while sequence-aware recommenders exist, they typically use recurrent neural networks or Transformer-based models, not classical DP.

**Differentiating our approach:** This confirmed the gap our project addresses. Existing GenAI-describable solutions for "LCS-based recommendation" were either purely theoretical or used LCS as a preprocessing step for neural models. Our contribution is a complete, standalone recommendation pipeline built directly on normalized LCS similarity — including a weighted scoring function, leave-one-out evaluation, and comparison against a baseline — without any learned model components. This makes the algorithm's contribution fully transparent and directly measurable.

We refined the approach further by asking: *"What are the limitations of using exact LCS match for item sequences in a recommendation context?"* This led us to incorporate a rating filter (keeping only positive interactions ≥ 4.0), a recency weight in the scoring function, and a popularity prior as a fallback — all design choices grounded in addressing weaknesses the AI identified but that we implemented ourselves from scratch.

All source code was written by the team members independently. No code was copied from AI output.

### 2.2 Dataset and Data Collection

**Dataset:** MovieLens ml-latest-small [1], a publicly available benchmark dataset maintained by the GroupLens research lab at the University of Minnesota. It contains 100,836 ratings applied to 9,742 movies by 610 users, collected between 1996 and 2018. Each rating record contains: `userId`, `movieId`, `rating` (0.5–5.0 scale), and `timestamp` (Unix epoch).

**Preprocessing pipeline (data_processing.py — Soonbee Hwang):**

The goal of preprocessing is to construct, for each user, an ordered sequence of movie IDs that represents their positive viewing history and is suitable for LCS comparison.

1. **Rating filter:** Only ratings ≥ 4.0 are retained. Ratings below this threshold are not reliable evidence that a user enjoyed a film; including them would introduce noise. This step reduces the dataset from 100,836 to approximately 28,000 ratings.

2. **Chronological sorting:** Each user's retained ratings are sorted by timestamp in ascending order. This is essential: LCS compares subsequences, so the ordering must reflect the actual temporal sequence of viewing decisions.

3. **Minimum activity filter:** Users with fewer than 10 positive ratings are excluded. Very short sequences produce unreliable similarity estimates.

4. **Sequence truncation:** Each user's sequence is truncated to the most recent 50 items. Very long historical sequences introduce old behavioral patterns that are less predictive of current preferences. Empirically, limiting to recent history improved recommendation quality.

5. **User selection:** The top 100 most active users (by sequence length after truncation) are selected for the experiment, to keep pairwise LCS computation feasible.

6. **Train/test split (leave-one-out):** The last item in each user's sequence is held out as the test target. The remaining items form the training sequence. This is the standard evaluation protocol for next-item prediction: the model is asked to rank the held-out item against all other items in the catalog.

After preprocessing, the experimental dataset consists of 100 users, each with 10–50 items in their training sequence (average approximately 35 items).

### 2.3 Algorithm: Longest Common Subsequence

The LCS algorithm finds the longest subsequence common to two sequences, where a subsequence preserves relative order but need not be contiguous. The algorithm is described in CLRS Chapter 15 [2].

**Recurrence relation:**

Given sequences X = ⟨x₁, x₂, …, xₘ⟩ and Y = ⟨y₁, y₂, …, yₙ⟩, define L[i][j] as the length of the LCS of the first i elements of X and the first j elements of Y:

```
L[i][j] = 0                              if i = 0 or j = 0
L[i][j] = L[i−1][j−1] + 1               if xᵢ = yⱼ
L[i][j] = max(L[i−1][j], L[i][j−1])     otherwise
```

The LCS length is given by L[m][n].

**Pseudocode — Space-Optimized LCS Length (lcs_length):**

```
function LCS_LENGTH(X, Y):
    // Ensure X is the longer sequence for memory efficiency
    if |X| < |Y|:
        swap X and Y
    m ← |X|,  n ← |Y|
    if n = 0: return 0

    prev ← array of (n+1) zeros    // previous row of DP table
    cur  ← array of (n+1) zeros    // current row

    for i from 1 to m:
        for j from 1 to n:
            if X[i−1] = Y[j−1]:
                cur[j] ← prev[j−1] + 1
            else:
                cur[j] ← max(prev[j], cur[j−1])
        swap prev and cur
        reset cur to all zeros

    return prev[n]
```

This space-optimized version uses only two rows instead of the full (m+1)×(n+1) table. Since we only need the previous row to compute the current row, we alternate between two arrays of size n+1.

- **Time complexity:** O(m × n), where m and n are the lengths of the two sequences.
- **Space complexity:** O(min(m, n)) — we always put the shorter sequence as Y, so both arrays have length proportional to the shorter sequence.

**Pseudocode — Full DP Table (lcs_length_full_table):**

When we also need to recover the actual LCS (not just its length), we must keep the full table:

```
function LCS_FULL_TABLE(X, Y):
    m ← |X|,  n ← |Y|
    L ← (m+1) × (n+1) matrix of zeros

    for i from 1 to m:
        for j from 1 to n:
            if X[i−1] = Y[j−1]:
                L[i][j] ← L[i−1][j−1] + 1
            else:
                L[i][j] ← max(L[i−1][j], L[i][j−1])

    return L   // LCS length is L[m][n]
```

- **Time complexity:** O(m × n)
- **Space complexity:** O(m × n)

**Pseudocode — Backtracking to Recover LCS (recover_lcs):**

```
function RECOVER_LCS(X, Y, L):
    i ← |X|,  j ← |Y|
    result ← empty list

    while i > 0 and j > 0:
        if X[i−1] = Y[j−1]:
            prepend X[i−1] to result
            i ← i−1,  j ← j−1
        else if L[i−1][j] ≥ L[i][j−1]:
            i ← i−1
        else:
            j ← j−1

    return result
```

The backtracking starts at L[m][n] and traces backward. At each cell: if the two elements matched, it contributed to the LCS; otherwise we move toward the larger of the two adjacent cells (up or left).

- **Time complexity:** O(m + n) for the backtracking pass.

### 2.4 Similarity Metric and Similarity Matrix

**Normalized LCS Similarity:**

Raw LCS length is not directly comparable across user pairs of different sequence lengths. We normalize:

```
similarity(X, Y) = LCS_LENGTH(X, Y) / max(|X|, |Y|)
```

This produces a score in [0, 1]. Two identical sequences score 1.0; two sequences with no common elements score 0.0.

**All-Pairs Similarity Matrix:**

```
function COMPUTE_SIMILARITY_MATRIX(sequences):
    // sequences: dictionary mapping userId → sequence
    user_ids ← list of all keys in sequences
    n ← |user_ids|
    sim_matrix ← n × n matrix of zeros

    for i from 0 to n−1:
        sim_matrix[i][i] ← 1.0
        for j from i+1 to n−1:
            sim ← NORMALIZED_LCS_SIMILARITY(sequences[user_ids[i]], sequences[user_ids[j]])
            sim_matrix[i][j] ← sim
            sim_matrix[j][i] ← sim    // matrix is symmetric

    return sim_matrix, user_ids
```

- **Time complexity:** O(n² × m²) for n users each with sequences of length at most m. For n=100 users and m=50 items, this is 100² × 50² / 2 ≈ 12.5M operations (upper triangle only). In practice, the computation completes in a few seconds.

### 2.5 Recommendation Engine

**Scoring Function:**

For a target user u and a candidate item i (seen by neighbor v but not by u), the recommendation score is:

```
score(u, i) = [Σ_{v ∈ neighbors} sim(u,v)^α × (1 + β × recency_v(i)) × (1 + γ × rating_v(i))]
              ──────────────────────────────────────────────────────────────────────────────────
                            [Σ_{v ∈ neighbors} sim(u,v)^α + ε]
              + λ × popularity(i)
```

where:
- **α = 1.0** (similarity power: how strongly neighbor similarity influences the vote)
- **β = 0.2** (recency weight: items watched more recently by a neighbor receive a boost)
- **γ = 0.5** (rating weight: items a neighbor rated highly receive a boost)
- **λ = 0.05** (popularity prior: a small fallback signal from item frequency)
- **ε** is a small epsilon to prevent division by zero

**Recency** of item i for neighbor v is computed as (position of i in v's sequence) / (sequence length of v), normalized to [0, 1]. A higher position index (later in the sequence) indicates more recent viewing.

**Popularity** is the log-scaled normalized frequency of item i across all training sequences.

**Pseudocode — Recommendation for One User:**

```
function RECOMMEND_FOR_USER(user_id, sim_matrix, train_sequences, train_interactions,
                             top_k_users, top_n_items, α, β, γ, λ):
    neighbors ← top_k_users most similar users to user_id (by sim_matrix)
    user_history ← set of items already seen by user_id
    item_scores ← empty dictionary

    for each (neighbor_id, sim_score) in neighbors:
        neighbor_seq ← train_interactions[neighbor_id]   // (movieId, rating) pairs
        seq_len ← |neighbor_seq|
        for position, (item, rating) in enumerate(neighbor_seq):
            if item in user_history: continue
            recency ← position / seq_len
            contribution ← sim_score^α × (1 + β × recency) × (1 + γ × rating)
            item_scores[item] += contribution

    total_sim ← Σ sim_score^α for all neighbors + ε
    for each item in item_scores:
        item_scores[item] ← item_scores[item] / total_sim + λ × popularity(item)

    if item_scores is empty:
        // fallback: recommend most popular unseen items
        return top_n_items most popular items not in user_history

    return top top_n_items items sorted by item_scores descending
```

### 2.6 Library Functions Used

| Library | Function(s) Used | Purpose |
|---------|-----------------|---------|
| **numpy** ≥ 1.26.0 [3] | `np.zeros`, `np.ndarray`, `np.triu_indices`, `np.argmax` | DP table allocation and matrix operations for the similarity computation |
| **pandas** ≥ 2.0.0 [4] | `pd.read_csv`, `DataFrame.groupby`, `DataFrame.sort_values` | Loading MovieLens CSV files and grouping/sorting ratings by user and timestamp |
| **matplotlib** ≥ 3.7.0 [5] | `plt.bar`, `plt.plot`, `plt.imshow`, `plt.table` | Generating all six result figures (hit rate charts, similarity heatmap, metrics table) |
| **seaborn** ≥ 0.12.0 [6] | `sns.heatmap` | Enhanced heatmap rendering with color bars for the user-user similarity matrix |
| **streamlit** ≥ 1.30.0 [7] | `st.selectbox`, `st.tabs`, `st.dataframe`, `st.pyplot` | Interactive web frontend for browsing recommendations and visualizing the DP table |
| **pytest** ≥ 7.0.0 [8] | `pytest` test runner | Unit testing framework for the 13 LCS algorithm tests |

### 2.7 Iterations and Improvements

The system went through several rounds of improvement from an initial prototype to the final version.

**Iteration 1 — Baseline LCS Vote:**
The initial implementation used a simple unweighted vote: for each candidate item, sum the LCS similarity scores of all neighbors who had watched it. Items unseen by any neighbor were scored zero. The random baseline beat this version for certain evaluation cutoffs, suggesting the raw similarity signal alone was insufficient.

**Iteration 2 — Rating Filter (≥ 4.0):**
A key insight was that treating all rated movies as positive feedback introduced noise. A user who rated a movie 2/5 likely did not enjoy it; including it in their sequence contaminated the similarity signal. Filtering to ratings ≥ 4.0 significantly improved Hit Rate, confirming that sequence quality matters more than sequence length.

**Iteration 3 — Sequence Truncation to Recent 50 Items:**
Early experiments used full viewing histories (some users have 200+ ratings). Using the most recent 50 items improved results, consistent with the intuition that recent behavior is more predictive of next preferences than older history.

**Iteration 4 — Rating-Aware and Recency-Aware Scoring:**
The scoring function was extended to weight each candidate item by how highly a neighbor rated it (γ = 0.5) and how recently the neighbor watched it (β = 0.2). This gave more importance to items strongly endorsed by similar users and to current behavioral trends rather than distant historical choices.

**Iteration 5 — Popularity Prior:**
A small popularity component (λ = 0.05) was added as a fallback signal. When neighborhood evidence is sparse (few neighbors have seen many relevant unseen items), the popularity prior stabilizes scores without dominating the similarity-based signal. The low weight ensures it acts as a tiebreaker, not an override.

### 2.8 Testing

Unit tests are in `tests/test_lcs.py`. All 13 tests pass.

| Test | Input | Expected Output | Result |
|------|-------|----------------|--------|
| `test_lcs_length_basic` | X=[1,2,3,4], Y=[1,3,4] | 3 | Pass |
| `test_lcs_length_empty` | X=[], Y=[1,2,3] | 0 | Pass |
| `test_lcs_length_identical` | X=Y=[10,20,30] | 3 | Pass |
| `test_lcs_length_no_common` | X=[1,2], Y=[3,4] | 0 | Pass |
| `test_full_table_matches_length` | X=[1,2,3,4,5], Y=[2,4,5] | L[5][3] = lcs_length(X,Y) = 3 | Pass |
| `test_recover_lcs` | X=[1,3,4,5,7], Y=[1,4,5,7] | [1,4,5,7] | Pass |
| `test_normalized_similarity_range` | X=[1,2,3], Y=[2,3,4] | sim ∈ [0.0, 1.0] | Pass |
| `test_normalized_similarity_identical` | X=Y=[1,2,3,4] | 1.0 | Pass |
| `test_normalized_similarity_empty` | X=[], Y=[1,2] | 0.0 | Pass |
| `test_similarity_matrix_shape` | 3 users, 3 sequences | matrix shape (3,3) | Pass |
| `test_similarity_matrix_diagonal` | 2 users | diagonal entries = 1.0 | Pass |
| `test_similarity_matrix_symmetric` | 3 users | sim[i][j] = sim[j][i] | Pass |
| `test_find_top_k_similar` | User 1 identical to User 2, dissimilar to User 3 | top neighbor is User 2 with score 1.0 | Pass |

Additionally, `recommendation.py` includes a **synthetic smoke test** using a manually constructed 3-user dataset with known ground truth, verifying that the end-to-end recommendation pipeline produces non-empty output and that the LCS-based model produces a hit for the synthetic test case.

To run all tests:
```
pytest tests/test_lcs.py -v
```

### 2.9 Experimental Results

The experiment was run on 100 users from the filtered MovieLens dataset, using a leave-one-out evaluation protocol (one held-out test item per user). The recommendation list length was N = 20 items.

**Table 1: Evaluation Metrics — LCS System vs. Random Baseline**

| Metric | K=1 | K=5 | K=10 | K=20 |
|--------|-----|-----|------|------|
| Hit Rate (LCS) | 0.0200 | 0.0300 | 0.0500 | 0.0500 |
| Hit Rate (Random) | 0.0000 | 0.0000 | 0.0000 | 0.0100 |
| NDCG (LCS) | — | — | 0.0304 | — |
| NDCG (Random) | — | — | ~0.000 | — |
| **MRR (LCS)** | **0.0248** | — | — | — |
| **MRR (Random)** | **0.0007** | — | — | — |

The LCS-based system **consistently and substantially outperforms the random baseline** across all metrics and all cutoffs. At K=20, LCS achieves a Hit Rate 5× higher than random (0.0500 vs. 0.0100). The MRR gap is even larger: 0.0248 vs. 0.0007, representing approximately a 35× improvement, indicating that when LCS makes a hit, it ranks the correct item higher in the list than random does.

**Discussion of results:**

The absolute Hit Rate values (around 5%) may initially appear modest. However, this reflects the inherent difficulty of next-item prediction: the model must identify 1 specific movie (out of nearly 9,000 in the MovieLens catalog) from a recommendation list of only 20 items. Viewed as a ranking task over 9,000 candidates, placing the correct item in the top-20 (top 0.2%) is a meaningful result.

The performance plateau between K=10 and K=20 is noteworthy: the system does not gain additional hits when the list is extended from 10 to 20 items. This suggests the signal from the 5 nearest neighbors is largely exhausted within the first 10 recommendations, and adding more items mostly adds noise or popular items that don't align with the specific test target.

The high MRR improvement over random is perhaps the most informative result: it shows that the LCS-based model not only finds the relevant item more often, but also ranks it more highly when it does find it. This is the behavior expected of a system that genuinely captures behavioral similarity, rather than making lucky random guesses.

A potential avenue for improvement is increasing neighborhood size beyond K=5 — our current setting — which may be too small to capture diverse item coverage. Grid search over hyperparameters (α, β, γ, λ) could also yield further improvements.

---

## 3. Conclusion

### 3.1 Answer to the Research Question

*Can sequence similarity measured by the LCS algorithm accurately predict a user's next content choice?*

Yes, to a meaningful degree. Our LCS-based recommendation system achieves a Hit Rate@20 of 5% and an MRR of 0.0248, compared to 1% and 0.0007 for a random baseline — improvements of 5× and ~35× respectively. The system demonstrates that preserving the temporal order of user interactions, and measuring behavioral similarity through sequence alignment, provides genuine predictive signal for next-item recommendation.

That said, the absolute performance is modest. LCS-based similarity is limited to exact item overlap; it cannot recognize that two users with similar genre preferences but disjoint movie catalogs are behaviorally similar. Incorporating content features (genres, directors, actors) alongside sequence similarity is the most promising direction for improvement.

### 3.2 Limitations

1. **Computational cost:** All-pairs LCS computation is O(n² × m²), which becomes prohibitive for large user bases. Scaling to millions of users would require approximate nearest-neighbor methods or pre-filtering.

2. **Exact item matching:** LCS requires the same movie ID to appear in both sequences to contribute to similarity. Genre-aware or embedding-based sequence matching could capture semantic similarity that exact LCS misses.

3. **Sparse evaluation:** Leave-one-out with a single test item per user is conservative. A user may have multiple reasonable next items; a system that recommends any of them should receive partial credit.

4. **Cold start:** The system cannot make LCS-based recommendations for new users with very short histories, falling back entirely to popularity.

5. **Static similarity:** The similarity matrix is computed once at evaluation time. In a production system, it would need to be recomputed incrementally as users accumulate new interactions.

### 3.3 Future Work

- **Windowed LCS:** Compute LCS only over a recent window of activity (e.g., last 20 items) to reduce computation and focus on current behavioral trends.
- **Semantic item representations:** Map movie IDs to genre or embedding vectors and measure sequence similarity at the semantic level.
- **Stronger baselines:** Compare against item-based collaborative filtering, BPR matrix factorization, or simple popularity-based recommendation to better contextualize LCS performance.
- **Hyperparameter tuning:** Systematic grid search over α, β, γ, λ and neighborhood size.
- **Parallel computation:** The similarity matrix can be computed in parallel across user pairs, which would make the approach more practical for larger datasets.

### 3.4 Individual Reflections

**Rebecca Wang:** This project gave me a concrete answer to a question I had wondered about since we first studied LCS in class: does the textbook application (DNA alignment, diff tools) exhaust the algorithm's usefulness, or can it do more? Applying it to user behavior showed me that the right framing transforms a standard algorithm into something novel. On the implementation side, working through the space optimization from O(mn) to O(min(m,n)) memory was a satisfying exercise in understanding DP at a deeper level than just getting the correct answer. I expect to use the skill of identifying "which part of the DP state do I actually need to keep?" in future systems work.

**Soonbee Hwang:** [Please fill in your personal reflection here — e.g., what you learned about data engineering, preprocessing, the importance of data quality for algorithm performance, or how this project connects to your broader goals.]

**Yue Liang:** [Please fill in your personal reflection here — e.g., what you learned about evaluation design, the gap between algorithmic correctness and practical utility, or how working on the scoring function and metrics shaped your thinking about recommendation systems.]

---

## References

[1] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4. https://doi.org/10.1145/2827872

[2] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2022. *Introduction to Algorithms*, 4th edition. MIT Press. Chapter 15: Dynamic Programming, Section 15.4: Longest Common Subsequence.

[3] Charles R. Harris et al. 2020. Array programming with NumPy. *Nature* 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2

[4] Wes McKinney. 2010. Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*. https://pandas.pydata.org

[5] J. D. Hunter. 2007. Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering* 9, 3, 90–95.

[6] Michael Waskom. 2021. seaborn: statistical data visualization. *Journal of Open Source Software* 6, 60, 3021. https://doi.org/10.21105/joss.03021

[7] Streamlit Inc. 2023. Streamlit — A faster way to build and share data apps. https://streamlit.io

[8] Krekel, H. et al. 2004. pytest — A mature full-featured Python testing framework. https://pytest.org

---

## Appendix: Source Code Files

The following source code files are included in the submitted ZIP archive:

| File | Description |
|------|-------------|
| `src/lcs_algo.py` | Core LCS dynamic programming algorithm and similarity utilities (Rebecca Wang) |
| `src/data_processing.py` | MovieLens data loading, sequence construction, and train/test split (Soonbee Hwang) |
| `src/recommendation.py` | Recommendation engine, weighted scoring function, and evaluation metrics (Yue Liang) |
| `src/visualize_results.py` | Result visualization — six figures (Yue Liang) |
| `src/main.py` | End-to-end pipeline: data → similarity → recommendations → evaluation → figures |
| `src/app.py` | Streamlit interactive web frontend |
| `tests/test_lcs.py` | 13 unit tests for the LCS algorithm (all passing) |

To run the full pipeline:
```
python src/main.py
```

To run unit tests:
```
pytest tests/test_lcs.py -v
```

To launch the interactive Streamlit app:
```
streamlit run src/app.py
```
