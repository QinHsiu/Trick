"""
    author: Qinhsiu
    date: 2022/11/13
"""

# 均匀分布
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def uniform_distribution():
    # for continuous
    a = 0
    b = 50
    size = 5000

    X_continuous = np.linspace(a, b, size)
    continuous_uniform = stats.uniform(loc=a, scale=b)
    continuous_uniform_pdf = continuous_uniform.pdf(X_continuous)

    # for discrete
    X_discrete = np.arange(1, 7)
    discrete_uniform = stats.randint(1, 7)
    discrete_uniform_pmf = discrete_uniform.pmf(X_discrete)

    # plot both tables
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # discrete plot
    ax[0].bar(X_discrete, discrete_uniform_pmf)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Probability")
    ax[0].set_title("Discrete Uniform Distribution")
    # continuous plot
    ax[1].plot(X_continuous, continuous_uniform_pdf)
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Probability")
    ax[1].set_title("Continuous Uniform Distribution")
    plt.savefig("./pic/uniform_distribution.png")

def gaussian_distribution():
    mu = 0
    variance = 1
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    plt.subplots(figsize=(8, 5))
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.title("Normal Distribution")
    plt.savefig("./pic/gaussian_distribution.png")

def log_normal_distribution():
    X = np.linspace(0, 6, 500)

    std = 1
    mean = 0
    lognorm_distribution = stats.lognorm([std], loc=mean)
    lognorm_distribution_pdf = lognorm_distribution.pdf(X)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(X, lognorm_distribution_pdf, label="μ=0, σ=1")
    ax.set_xticks(np.arange(min(X), max(X)))

    std = 0.5
    mean = 0
    lognorm_distribution = stats.lognorm([std], loc=mean)
    lognorm_distribution_pdf = lognorm_distribution.pdf(X)
    plt.plot(X, lognorm_distribution_pdf, label="μ=0, σ=0.5")

    std = 1.5
    mean = 1
    lognorm_distribution = stats.lognorm([std], loc=mean)
    lognorm_distribution_pdf = lognorm_distribution.pdf(X)
    plt.plot(X, lognorm_distribution_pdf, label="μ=1, σ=1.5")

    plt.title("Lognormal Distribution")
    plt.legend()
    plt.savefig("./pic/log_normal_distribution.png")

def poisson_distribution():
    from scipy import stats

    print(stats.poisson.pmf(k=9, mu=3))
    X = stats.poisson.rvs(mu=3, size=500)

    plt.subplots(figsize=(8, 5))
    plt.hist(X, density=True, edgecolor="black")
    plt.title("Poisson Distribution")
    plt.savefig("./pic/poisson_distribution.png")


def exponent_distribution():
    X = np.linspace(0, 5, 5000)

    exponetial_distribtuion = stats.expon.pdf(X, loc=0, scale=1)

    plt.subplots(figsize=(8, 5))
    plt.plot(X, exponetial_distribtuion)
    plt.title("Exponential Distribution")
    plt.savefig("./pic/exponent_distribution.png")

def binomial_distribution():
    X = np.random.binomial(n=1, p=0.5, size=1000)

    plt.subplots(figsize=(8, 5))
    plt.hist(X)
    plt.title("Binomial Distribution")
    plt.savefig("./pic/binomial_distribution.png")


def tau_distribution():
    import seaborn as sns
    from scipy import stats

    X1 = stats.t.rvs(df=1, size=4)
    X2 = stats.t.rvs(df=3, size=4)
    X3 = stats.t.rvs(df=9, size=4)

    plt.subplots(figsize=(8, 5))
    sns.kdeplot(X1, label="1 d.o.f")
    sns.kdeplot(X2, label="3 d.o.f")
    sns.kdeplot(X3, label="6 d.o.f")
    plt.title("Student's t distribution")
    plt.legend()
    plt.savefig("./pic/tau_distribution.png")

def cardinal_distribution():
    X = np.arange(0, 6, 0.25)

    plt.subplots(figsize=(8, 5))
    plt.plot(X, stats.chi2.pdf(X, df=1), label="1 d.o.f")
    plt.plot(X, stats.chi2.pdf(X, df=2), label="2 d.o.f")
    plt.plot(X, stats.chi2.pdf(X, df=3), label="3 d.o.f")
    plt.title("Chi-squared Distribution")
    plt.legend()
    plt.savefig("./pic/cardinal_distribution.png")

if __name__ == '__main__':
    uniform_distribution()
    log_normal_distribution()
    gaussian_distribution()
    binomial_distribution()
    poisson_distribution()
    cardinal_distribution()
    exponent_distribution()
    tau_distribution()

