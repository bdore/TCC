# AGGRAWAL ET AL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


N = 10000000
v_1 = 0.01
v_2 = 0.2
t = np.array(np.arange(1, 41))


def model(N, t, v_1, v_2):
    numerator = 1 - np.exp(-(v_1 + v_2) * t)
    denominator = 1 + (v_2 / v_1) * np.exp(-(v_1 + v_2) * t)
    v_t = N * (numerator/denominator)
    return pd.Series(v_t)


def fit_log(x, a, b):
    return a * np.log(x) + b


v_t = model(N, t, v_1, v_2)
ax = v_t.plot(x=t, color='blue')
ax.ticklabel_format(style='scientific')
ax.set_xlabel("t", fontsize=14)
ax.set_ylabel("V(t)", fontsize=14)
plt.show()

x = np.array(t[1:], dtype=float)
y = np.log(np.array(v_t[1:], dtype=float))
popt, pcov = curve_fit(fit_log, x, y)
exp_model_data_test = fit_log(x, *popt)
r2 = np.round_(r2_score(y, exp_model_data_test), 3)
ln_v_t = np.log(v_t)
log_model = pd.Series(fit_log(t, popt[0], popt[1]))

ax = log_model.plot(
    x=t, color='orange', label='$f(t), R^2 = ${}'.format(r2), legend=True, linewidth=2
)
ax = ln_v_t.plot(
    x=t, color='blue', label='$ln(V(t))$', legend=True, figsize=(7,6), linewidth=2
)
ax.ticklabel_format(style='plain')
ax.legend(fontsize=12)
ax.set_xlabel("$t$", fontsize=14)
ax.set_ylabel("$y$", fontsize=14)
plt.show()
