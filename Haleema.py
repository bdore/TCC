import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

buyer = [201.58, 207.02, 217.66, 233.49, 252.59, 271.49, 286.49, 295.53, 299.23, 300.00]
seller = [397.01, 387.48, 369.72, 344.51, 315.42, 287.97, 267.26, 255.45, 250.89, 250.00]

df = pd.DataFrame(data={'buyer': buyer, 'seller': seller})

# BUYER
y = df['buyer'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(y)
y = scaler.transform(y).flatten()

x = df.index.values.reshape(-1, 1)
x_scaler = MinMaxScaler(feature_range=(-15, 15))
x_scaler.fit(x)
x = x_scaler.transform(x).flatten()


def fit_logistic(x, a, b):
    return 1 / (1 + np.exp(a - b * x))


popt, pcov = curve_fit(fit_logistic, x, y, p0=[0, 0], bounds=(-1, 1))


def fit_logistic_delta(x, a, b, c):
    x_scaler = MinMaxScaler(feature_range=(-15, 15))
    x_scaler.fit(x.reshape(-1, 1))
    x = x_scaler.transform(x.reshape(-1, 1)).flatten()
    return c + 1 / 2 * c * (1 / (1 + np.exp(a - b * x)))


starting_price = 200
x = df.index.values
y_hat = fit_logistic_delta(x, popt[0], popt[1], starting_price)
original = scaler.inverse_transform(y.reshape(-1, 1))
model = y_hat.reshape(-1, 1)
results = pd.DataFrame(data={'original': original.flatten(), 'model': model.flatten()})

r2 = np.round_(r2_score(results.original, results.model), 4)
plt.figure(figsize=(7, 5))
for column in results.columns:
    if column == 'model':
        name = 'Este\/estudo'
        plt.plot(
            results.index.values, results[column], color='blue',
            label='${}, R^2={}$'.format(name, r2)
        )
    else:
        plt.plot(results.index.values, results[column], color='grey', label='$Haleema\/2018$')
plt.xlabel('Rodada de negociação', fontsize=15)
plt.ylabel('Oferta de compra', fontsize=15)
plt.legend()
plt.show()

# SELLER
seller_y_scaler = MinMaxScaler()
y = df['seller'].values.reshape(-1, 1)
seller_y_scaler.fit(y)
y = seller_y_scaler.transform(y).flatten()


def fit_logistic_seller(x, a, b):
    return (-1 * (1 / (1 + np.exp(a - b * x)))) + 1


x = df.index.values

x_scaler = MinMaxScaler(feature_range=(-7, 18))
x_scaler.fit(x.reshape(-1, 1))
x = x_scaler.transform(x.reshape(-1, 1)).flatten()

popt, pcov = curve_fit(fit_logistic_seller, x, y, p0=[0, 0], bounds=(-1, 1.5))


def fit_logistic_seller_start(x, a, b, delta):
    return delta - (3 / 8) * delta * (1 / (1 + np.exp(a - b * x)))


y_hat = fit_logistic_seller_start(x, popt[0], popt[1], 400)

seller_df = pd.DataFrame(data={'original': df['seller'], 'model': y_hat.flatten()})

r2 = np.round_(r2_score(seller_df.original, seller_df.model), 4)
plt.figure(figsize=(7, 5))
for column in seller_df.columns:
    if column == 'model':
        name = 'Este\/estudo'
        plt.plot(
            seller_df.index.values, seller_df[column], color='blue', label='${}, R^2={}$'.format(name, r2)
        )
    else:
        plt.plot(seller_df.index.values, seller_df[column], color='grey', label='$Haleema\/2018$')
plt.xlabel('Rodada de negociação', fontsize=15)
plt.ylabel('Oferta de venda', fontsize=15)
plt.legend()
plt.show()


final_df = pd.DataFrame(data={
    'Vendedor, Este Estudo': seller_df.model,
    'Comprador, Este Estudo': results.model
})


# DELTA, X + ALPHA
def fit_logistic_seller_delta_alpha(x, a, b, delta, alpha):
    return delta - 3/8 * delta * (1 / (1 + np.exp(a - b * (x + alpha))))


# DELTA, H(X), ALPHA
def fit_logistic_seller_delta_alpha_h_x(x, a, b, delta, alpha):
    h_x = np.power(
        np.log(x_scaler.inverse_transform(x.reshape(-1,1)).flatten() + 1), 1/4
    ) * alpha * (delta * 0.0002)
    return delta - ((3 + h_x)/8) * delta * (1 / (1 + np.exp(a - b * (x))))


# DELTA, H(X), X + ALPHA
def fit_logistic_seller_delta_alpha_h_x_x_plus_alpha(x, a, b, delta, alpha):
    h_x = np.power(
        np.log(x_scaler.inverse_transform(x.reshape(-1,1)).flatten() + 1), 1/4
    ) * alpha * (delta * 0.0002)
    return delta - ((3 + h_x)/8) * delta * (1 / (1 + np.exp(a - b * (x + alpha))))


for func in [fit_logistic_seller_delta_alpha, fit_logistic_seller_delta_alpha_h_x,
             fit_logistic_seller_delta_alpha_h_x_x_plus_alpha]:
    alpha = [2, 4, 8]
    for a in alpha:
        y_hat = func(x, popt[0], popt[1], 400, a)
        final_df['{}'.format(a)] = y_hat

    plt.figure(figsize=(7, 5))
    n = 5
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    for column, color in zip(final_df.columns, colors):
        if column in ['2', '4', '8']:
            plt.plot(
                final_df.index.values, final_df[column], color=color,
                label=r'$\alpha={}$'.format(column)
            )
        else:
            plt.plot(
                final_df.index.values, final_df[column], color=color,
                label='${}$'.format(column)
            )
    plt.xlabel('Rodada', fontsize=15)
    plt.ylabel('Oferta', fontsize=15)
    plt.legend()
    plt.show()
