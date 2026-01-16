import pandas as pd

import numpy as np

import sys

WORKING_BOOK = 35
if len(sys.argv) > 1:
    WORKING_BOOK = int(sys.argv[1])

# ---------- Load similarity matrix ----------
sim_df = pd.read_csv("sim_dfx.csv", index_col=0)
sim_df.index = sim_df.index.astype(int)
sim_df.columns = sim_df.columns.astype(int)

# (Optional) verify index/columns align
# assert sim_df.index.equals(sim_df.columns), "Row/column labels differ!"

# ---------- Parameters ----------
K = 20  # number of recommendations per book

# ---------- Compute top-k neighbors per row ----------
vals = sim_df.values.copy()

# Exclude self-similarities so they don't appear in top-k
# We set the diagonal to -inf so it won't be selected
np.fill_diagonal(vals, -np.inf)

# For each row, find indices of the top-K largest values without full sort
# argpartition gives K largest in arbitrary order; we will sort those K.
topk_idx = np.argpartition(-vals, kth=K-1, axis=1)[:, :K]  # shape: (n_rows, K)

# Now sort those K by actual similarity descending for each row
row_indices = np.arange(vals.shape[0])[:, None]
topk_vals = vals[row_indices, topk_idx]  # shape: (n_rows, K)
order_within_k = np.argsort(-topk_vals, axis=1)
topk_idx_sorted = topk_idx[row_indices, order_within_k]   # neighbor column indices sorted by sim desc

# Map column indices to actual book IDs
book_ids = sim_df.index.to_numpy()
col_ids = sim_df.columns.to_numpy()
topk_neighbor_ids = col_ids[topk_idx_sorted]  # shape: (n_rows, K)

# ---------- Build wide DataFrame: book_id + choice_1..choice_K ----------
choice_cols = {f"choice_{i+1}": topk_neighbor_ids[:, i] for i in range(K)}
wide_df = pd.DataFrame({"book_id": book_ids, **choice_cols})

# Write to CSV
wide_out_path = "book_topk_wide.csv"
wide_df.to_csv(wide_out_path, index=False)
print(f"Wrote wide CSV: {wide_out_path} (shape={wide_df.shape})")
