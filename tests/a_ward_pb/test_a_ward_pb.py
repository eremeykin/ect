import numpy as np

from clustering.agglomerative.a_ward import AWard
from tests.tools import transformation_exists
from clustering.agglomerative.a_ward_pb import AWardPB

def test_a_ward_pb(data_cs_k_star_res):
    cs = data_cs_k_star_res.cs
    k_star = data_cs_k_star_res.k_star
    actual = data_cs_k_star_res.res['labels']

    # run_a_ward = AWard(cs, k_star)
    # result = run_a_ward()

    run_a_ward_pb = AWardPB(cs, k_star)
    result2 = run_a_ward_pb()

    # assert transformation_exists(result2, result)
    assert transformation_exists(actual, result2)
