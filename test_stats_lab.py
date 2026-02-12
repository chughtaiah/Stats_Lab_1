import numpy as np
from stats_lab import (
    sample_mean,
    sample_variance,
    sample_covariance,
    covariance_matrix,
    order_statistics
)


def test_sample_mean():
    data = np.array([1, 2, 3, 4, 5])
    assert sample_mean(data) == 3


def test_sample_variance():
    data = np.array([1, 2, 3, 4, 5])
    assert np.isclose(sample_variance(data), 2.5)


def test_sample_covariance():
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    assert np.isclose(sample_covariance(x, y), 2)


def test_covariance_matrix():
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    cov_matrix = covariance_matrix(x, y)

    assert cov_matrix.shape == (2, 2)
    assert np.isclose(cov_matrix[0, 1], 2)


def test_order_statistics():
    data = np.array([5, 1, 3, 2, 4])
    minimum, maximum, median, q1, q3 = order_statistics(data)

    assert minimum == 1
    assert maximum == 5
    assert median == 3

