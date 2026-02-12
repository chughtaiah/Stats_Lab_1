import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent GUI issues

from stats_lab import (
    normal_histogram,
    uniform_histogram,
    bernoulli_histogram,
    sample_mean,
    sample_variance,
    order_statistics,
    sample_covariance,
    covariance_matrix
)

# -----------------------------------
# Q1 – Normal (10 pts)
# -----------------------------------
def test_q1_normal():
    np.random.seed(0)
    data = normal_histogram(10000)
    assert isinstance(data, (list, np.ndarray))
    data = np.asarray(data)
    assert abs(sample_mean(data)) < 0.1


# -----------------------------------
# Q1 – Uniform (10 pts)
# -----------------------------------
def test_q1_uniform():
    np.random.seed(0)
    data = uniform_histogram(10000)
    data = np.asarray(data)
    assert abs(sample_mean(data) - 5) < 0.1


# -----------------------------------
# Q1 – Bernoulli (10 pts)
# -----------------------------------
def test_q1_bernoulli():
    np.random.seed(0)
    data = bernoulli_histogram(10000)
    data = np.asarray(data)
    assert abs(sample_mean(data) - 0.5) < 0.05


# -----------------------------------
# Q2 – Mean (10 pts)
# -----------------------------------
def test_q2_mean():
    data = np.array([1, 2, 3, 4, 5])
    assert sample_mean(data) == 3


# -----------------------------------
# Q2 – Variance (10 pts)
# -----------------------------------
def test_q2_variance():
    data = np.array([1, 2, 3, 4, 5])
    assert np.isclose(sample_variance(data), 2.5)


# -----------------------------------
# Q3 – Order Statistics (20 pts)
# -----------------------------------
def test_q3_order_statistics():
    data = np.array([5, 1, 3, 2, 4])
    minimum, maximum, median, q1, q3 = order_statistics(data)

    assert minimum == 1
    assert maximum == 5
    assert median == 3
    assert q1 == 2
    assert q3 == 4


# -----------------------------------
# Q4 – Covariance (15 pts)
# -----------------------------------
def test_q4_covariance():
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    assert np.isclose(sample_covariance(x, y), 2)


# -----------------------------------
# Q5 – Covariance Matrix (15 pts)
# -----------------------------------
def test_q5_covariance_matrix():
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    cm = covariance_matrix(x, y)

    assert cm.shape == (2, 2)
    assert np.isclose(cm[0, 1], 2)
