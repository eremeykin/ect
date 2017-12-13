from clustering.divisive.bikm_r import BiKMeansR
from tests.tools import transformation_exists


def test_bikm_r(data_res):
    run_bikm_r = BiKMeansR(data_res.data, epsilon=0.32)
    result = run_bikm_r()
    assert transformation_exists(result, data_res.res)
