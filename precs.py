import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

WORKING_BOOK = 39

prod_df = pd.read_csv("ratings.csv").head(1000000)
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
sim_df.to_csv("sim_df.csv", index=True)  # keep index so you preserve book labels

# --- Load ---
sim_df = pd.read_csv("sim_df.csv", index_col=0)

# print(sim_array)
# print(matrix)


def most_similar_books(sim_df: pd.DataFrame, book_id: int, k: int=10):
    if book_id not in sim_df.index:
        raise KeyError("Book ID not found.")
    
    sims = sim_df.loc[book_id]

    top = sims.sort_values(ascending=False).head(k)

    return top.index.tolist()

def get_book_title(book_id: int):
    
    match = books_names_table.loc[
            books_names_table["book_id"] == book_id, "original_title"
        ]
    return match.iloc[0] if not match.empty else None

books_names_table = pd.read_csv("books.csv")
books_names_table = books_names_table[["book_id", "original_title"]]

outcome = most_similar_books(sim_df=sim_df,book_id=WORKING_BOOK, k=10)
print(outcome)

for item in outcome:
    print(get_book_title(item))
