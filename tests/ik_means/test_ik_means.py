from clustering.ik_means.ik_means import IKMeans
from tests.tools import transformation_exists


def test_ik_means(data_cs_res):
    run_ik_means = IKMeans(data_cs_res.cs)
    my_labels = run_ik_means()
    assert transformation_exists(data_cs_res.res['labels'], my_labels)
