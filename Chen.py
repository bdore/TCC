import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = np.median([0.0074,0.0046,0.0014,0.0137,0.0127,0.003,0.0065,0.0003,0.004,0.082])
e = np.median([0.71,0.55,0.38,0.65,0.95,0.43,0.94,0.18,0.68,0.97])

def model(x, a, b):
    return -a * (x ** 2) + b * x

x = np.arange(0, 120)
v_t = pd.Series(model(x, d, e))
ax = v_t.plot(figsize=(7, 6), color='blue')
ax.ticklabel_format(style='plain')
ax.set_xlabel("$T$", fontsize=14)
ax.set_ylabel("$W(T)$", fontsize=14)
plt.show()