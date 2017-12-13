from collections import namedtuple

import numpy as np
import pytest
from tests.tools import rp, matlab_connector

DataRes = namedtuple('DataRes', 'data res')


@pytest.fixture(params=[
    DataRes(rp("shared/data/5_clusters.pts"), 'labels'),
])
def data_res(request):
    param = request.param
    if param.res == 'labels':
        data = np.loadtxt(param.data)
        labels = np.loadtxt(param.data[:param.data.rfind('.')] + ".lbs")
        return DataRes(data, labels)
