# ZHAO ET AL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

N = 2000
N_2_pct = 0.01 # positive opinion leaders percentage in the population
N_3_pct = 0.29 # negative opinion leaders percentage in the population
N_2 = np.int16(N * N_2_pct)
N_3 = np.int16(N * N_3_pct)
N_1 = N - N_2 - N_3
z = w = 0.5

opinion_followers = np.arange(-0.5, 0.5001, 0.0001)
opinion_leaders_positive = np.arange(0.0001, 0.5001, 0.0001)
opinion_leaders_negative = np.arange(-0.5000, 0, 0.0001)

epsilon_f = np.arange(0, 1.01, 0.01)
epsilon_p = epsilon_n = 0.25

N_1_vec = np.random.choice(opinion_followers, N_1)
N_2_vec = np.random.choice(opinion_leaders_positive, N_2)
N_3_vec = np.random.choice(opinion_leaders_negative, N_3)

epsilon_f_vec = np.random.choice(epsilon_f, N_1)
epsilon_p_vec = np.array([epsilon_p] * N_2)
epsilon_n_vec = np.array([epsilon_n] * N_3)

w_vec = np.array([w] * N_2).reshape(-1,1)
z_vec = np.array([z] * N_3).reshape(-1,1)

def followers_opinions(N_1_vec,
                       N_2_vec,
                       N_3_vec,
                       epsilon_f_vec,
                       epsilon_p_vec,
                       epsilon_n_vec,
                       w_vec,
                       z_vec):
    alpha = beta = 0.4
    g = -0.5
    d = 0.5

    def updating_weight(x_i, x_j, epsilon_i):
        return np.abs((x_i.reshape(-1, 1) - x_j.reshape(1, -1))) <= epsilon_i.reshape(-1, 1)

    followers_a_ij = np.int8(updating_weight(N_1_vec, N_1_vec, epsilon_f_vec))
    alpha_a_ij = np.int8(updating_weight(N_1_vec, N_2_vec, epsilon_f_vec))
    beta_a_ij = np.int8(updating_weight(N_1_vec, N_3_vec, epsilon_f_vec))

    followers_a_ij_sum = followers_a_ij.sum(axis=1).reshape(-1, 1)
    followers_a_ij_sum = np.where(followers_a_ij_sum == 0, 0.000000001, followers_a_ij_sum)
    followers = (1 - alpha - beta) * (1 / followers_a_ij_sum) * (
        (followers_a_ij * N_1_vec).sum(axis=1).reshape(-1, 1)
    )

    alpha_a_ij_sum = alpha_a_ij.sum(axis=1).reshape(-1, 1)
    alpha_a_ij_sum = np.where(alpha_a_ij_sum == 0, 0.000000001, alpha_a_ij_sum)
    positive_leaders = alpha * (1 / alpha_a_ij_sum) * (
        (alpha_a_ij * N_2_vec).sum(axis=1).reshape(-1, 1)
    )

    beta_a_ij_sum = beta_a_ij.sum(axis=1).reshape(-1, 1)
    beta_a_ij_sum = np.where(beta_a_ij_sum == 0, 0.000000001, beta_a_ij_sum)
    negative_leaders = beta * (1 / beta_a_ij_sum) * (
        (beta_a_ij * N_3_vec).sum(axis=1).reshape(-1, 1)
    )

    opinions_followers = followers + positive_leaders + negative_leaders

    # POSITIVE OPINION LEADERS
    positive_leaders_a_ij = np.int8(updating_weight(N_2_vec, N_2_vec, epsilon_p_vec))
    positive_leaders_a_ij_sum = positive_leaders_a_ij.sum(axis=1).reshape(-1, 1)
    opinions_positive_leaders = (1 - w_vec) * (1 / positive_leaders_a_ij_sum) * (
        (positive_leaders_a_ij * N_2_vec).sum(axis=1).reshape(-1, 1)
    ) + w_vec * d

    # NEGATIVE OPINION LEADERS
    negative_leaders_a_ij = np.int8(updating_weight(N_3_vec, N_3_vec, epsilon_n_vec))
    negative_leaders_a_ij_sum = negative_leaders_a_ij.sum(axis=1).reshape(-1, 1)
    opinions_negative_leaders = (1 - z_vec) * (1 / negative_leaders_a_ij_sum) * (
        (negative_leaders_a_ij * N_3_vec).sum(axis=1).reshape(-1, 1)) + z_vec * g

    return opinions_followers, opinions_positive_leaders, opinions_negative_leaders


data_list = {'N_1_vec': [], 'N_2_vec': [], 'N_3_vec': []}
for i in range(0, 20):
    data_list['N_1_vec'].append(N_1_vec)
    data_list['N_2_vec'].append(N_2_vec)
    data_list['N_3_vec'].append(N_3_vec)
    N_1_vec, N_2_vec, N_3_vec = followers_opinions(
        N_1_vec, N_2_vec, N_3_vec, epsilon_f_vec, epsilon_p_vec, epsilon_n_vec, w_vec, z_vec
    )
    N_1_vec = N_1_vec.reshape(-1)
    N_2_vec = N_2_vec.reshape(-1)
    N_3_vec = N_3_vec.reshape(-1)

# PLOT 5x3 PANEL
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3)
plt.subplots_adjust(left=0.01,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.4)
axes = [[ax1, ax4, ax7, ax10, ax13], [ax2, ax5, ax8, ax11, ax14], [ax3, ax6, ax9, ax12, ax15]]

for agent, symbol, ax_list in zip(['N_1_vec', 'N_2_vec', 'N_3_vec'], ['F', 'P', 'N'], axes):
    for i, ax_ in zip([0,1,2,3,9], ax_list):
        b = pd.DataFrame(data_list[agent][i])
        b.rename({0: 'opinions'}, inplace=True, axis=1)
        b['i'] = b.index.values
        ax_.set_title('$x_i^{}(t), t={}$'.format(symbol, i), size=18)
        ax = b.plot.scatter(
            x='i', y='opinions', c='blue', alpha=0.2, ax=ax_, figsize=(17,25)
        )
        ax.set_xlabel('$x_i$', size=14)
        ax.set_ylabel('')

for ax_ylabel in axes[0]:
    ax_ylabel.set_ylabel('Opinião', size=22)

plt.show()
