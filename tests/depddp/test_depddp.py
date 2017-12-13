from clustering.divisive.depddp import DEPDDP
from tests.tools import transformation_exists


def test_depddp(data_res):
    run_depddp = DEPDDP(data_res.data)
    result = run_depddp()
    assert transformation_exists(result, data_res.res)
