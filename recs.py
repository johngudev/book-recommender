
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df= pd.DataFrame(columns=["user", "book","ratings"], data=[[1,1,5],[1,2,5],[1,3,1],
                                                           [2,1,2],[2,3,4],
                                                        #    [2,1,2],[2,2,2],[2,3,4],
                                                            # [2,1,2],[2,2,2],[2,3,4],
                                                           [3,1,4],[3,2,4],[3,3,2],
                                                           [1,4,3],[2,4,4],[3,4,2]])


# Pivot to matrix (users → rows, books → columns)
matrix = df.pivot(index="book", columns="user", values="ratings")
matrix = matrix.fillna(0)
print(matrix)

sim_array = cosine_similarity(matrix.values)


sim_df = pd.DataFrame(data=sim_array)

print(sim_array)
# print(matrix)