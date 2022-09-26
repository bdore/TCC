# WEISBUCH ET AL


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

opinions = np.arange(0, 1.0001, 0.0001)
epsilon = 0.25
mu = 0.3
x_i = np.random.choice(opinions, 1500)


def get_opinions(x_i):
    shuffled = np.random.permutation(x_i)
    pairs = np.array(np.array_split(shuffled, len(shuffled) / 2))

    def update_opinion(x_i, x_j):
        return x_i + mu * (x_j - x_i)

    updated_x_i = update_opinion(pairs[:, 0], pairs[:, 1]).reshape(-1, 1)
    updated_x_j = update_opinion(pairs[:, 1], pairs[:, 0]).reshape(-1, 1)
    updated_pairs = np.concatenate([updated_x_i, updated_x_j], axis=1)
    updated_opinions = np.where(
        np.abs(pairs[:, 0] - pairs[:, 1]).reshape(-1, 1) < epsilon, updated_pairs, pairs
    ).reshape(-1)
    return pairs[:, 0], pairs[:, 1], updated_x_i, updated_opinions


data = list()
for i in range(1000):
    x_i = get_opinions(x_i)[3]
    data.append(x_i)

data_df = pd.DataFrame(data)
new_df = data_df.transpose().unstack().reset_index()

figure = plt.figure(figsize=(10, 10))
for i in range(70):
    plt.scatter(
        x=new_df[new_df.level_0 == i].level_0, y=new_df[new_df.level_0 == i][0], alpha=0.05,
        c='blue', s=7
    )
plt.xlabel('$t$', fontsize=14)
plt.ylabel('$Opinião$', fontsize=14)
plt.show()
