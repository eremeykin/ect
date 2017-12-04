import numpy as np
from clustering.agglomerative.algorithms.a_ward import AWard
from tests.tools import transformation_exists


def test_a_ward(data_cs_k_star_res):
    cs = data_cs_k_star_res.cs
    k_star = data_cs_k_star_res.k_star
    actual = np.loadtxt(data_cs_k_star_res.res)

    run_a_ward = AWard(cs, k_star)
    result = run_a_ward()

    assert transformation_exists(actual, result)
