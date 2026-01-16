import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys

WORKING_BOOK = 35

if len(sys.argv) > 1:
    WORKING_BOOK = int(sys.argv[1])

sim_df = pd.read_csv("sim_dfx.csv", index_col=0)



sim_df.index = sim_df.index.astype(int)
sim_df.columns = sim_df.columns.astype(int)


def most_similar_books(sim_df: pd.DataFrame, book_id: int, k: int=10):
    if book_id not in sim_df.index:
        raise KeyError("Book ID not found.")
    
    sims = sim_df.loc[book_id]

    top = sims.sort_values(ascending=False).head(k)

    return top.index.tolist()

def get_book_title(book_id: int):
    
    match = books_names_table.loc[
            books_names_table["book_id"] == book_id, "title"
        ]
    return match.iloc[0] if not match.empty else None

books_names_table = pd.read_csv("books.csv")
books_names_table = books_names_table[["book_id", "title"]]

outcome = most_similar_books(sim_df=sim_df,book_id=WORKING_BOOK, k=30)
print(outcome)

for item in outcome:
    print(get_book_title(item))
