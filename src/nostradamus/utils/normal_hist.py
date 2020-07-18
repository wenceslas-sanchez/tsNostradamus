import numpy as np
from scipy.stats import skewnorm
import scipy
import matplotlib.pyplot as plt

def compare_hist_to_norm(data, bins=25):
    """

    :param data:
    :param bins:
    :return:
    """
    fig = plt.figure(figsize=(10, 5))

    mu, std = scipy.stats.norm.fit(data)

    plt.hist(data, bins=bins, density=True, alpha=0.6, color='purple', label="Données")

    # Plot le PDF.
    xmin, xmax = plt.xlim()
    X = np.linspace(xmin, xmax)

    plt.plot(X, scipy.stats.norm.pdf(X, mu, std), label="Normal Distribution")
    plt.plot(X, skewnorm.pdf(X, *skewnorm.fit(data)), color='black', label="Skewed Normal Distribution")

    mu, std = scipy.stats.norm.fit(data)
    sk = scipy.stats.skew(data)

    title2 = "Moments mu: {}, sig: {}, sk: {}".format(round(mu, 4), round(std, 4), round(sk, 4))
    plt.ylabel("Fréquence", rotation=90)
    plt.title(title2)
    plt.legend()
    #plt.show()
    pass



def compare_hist_to_norm_ax(x, data, bins=25):

    mu, std = scipy.stats.norm.fit(data)

    ax[x].hist(data, bins=bins, density=True, alpha=0.6, color='purple', label="Données")

    # Plot le PDF.
    xmin, xmax = plt.xlim()
    X = np.linspace(xmin, xmax)

    ax[x].plot(X, scipy.stats.norm.pdf(X, mu, std), label="Normal Distribution")
    ax[x].plot(X, skewnorm.pdf(X, *skewnorm.fit(data)), color='black', label="Skewed Normal Distribution")

    mu, std = scipy.stats.norm.fit(data)
    sk = scipy.stats.skew(data)

    title2 = "Moments mu: {}, sig: {}, sk: {}".format(round(mu, 4), round(std, 4), round(sk, 4))
    ax[x].ylabel("Fréquence", rotation=90)
    ax[x].title(title2)
    ax[x].legend()
    pass