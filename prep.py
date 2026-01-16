import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

WORKING_BOOK = 35

prod_df = pd.read_csv("ratings.csv")
print("")

print("Number of books: ")
print(len(prod_df["book"].unique()))
print("----------------------------------------------------")
print("Number of users: ")
print(len(prod_df["user"].unique()))
print("")

book_counts = (
    prod_df["book"]
    .value_counts()          # count entries per book
    .sort_values(ascending=False)
)

# Print top 10 books
print(book_counts.head(10).to_string())
print("")

# Pivot to matrix (users → rows, books → columns)
matrix = prod_df.pivot(index="book", columns="user", values="rating")
# matrix = matrix.fillna(0)
# print(matrix)

#>>>added
book_means = matrix.mean(axis=1)
matrix_centered = matrix.sub(book_means, axis=0).fillna(0)
sim_array = cosine_similarity(matrix_centered.values)
sim_df = pd.DataFrame(sim_array, index=matrix.index, columns=matrix.index)
#>>>added

sim_array = cosine_similarity(matrix_centered.values)


sim_df = pd.DataFrame(data=sim_array, index=matrix_centered.index, columns=matrix_centered.index)


# --- Save ---
sim_df.to_csv("sim_dfx.csv", index=True)  # keep index so you preserve book labels
