import subprocess
import re
import numpy as np
import os

MATLAB_PATH = "/home/eremeykin/Matlab2015b/bin/matlab"
SHARED_MATLAB = "/media/d_disk/projects/Clustering/ect/tests/shared/matlab"


def rp(path):
    return os.path.join(os.path.dirname(__file__), path)


def array_equals_up_to_order(array1, array2, atol=1.e-5):
    if array1.shape != array2.shape:
        return False
    # array1 = array1[array1[:, 0].argsort()]
    # array2 = array2[array2[:, 0].argsort()]
    deleted = set()
    for row1 in array1:
        def inner():
            for index, row2 in enumerate(array2):
                if index not in deleted:
                    if np.allclose(row1, row2, atol=atol):
                        deleted.add(index)
                        return True
        if not inner():
            return False
    return deleted == set(range(len(array1)))
    # return np.allclose(array1, array2, atol=atol)


def transformation_exists(X, Y):
    if len(X) != len(Y):
        return False
    XY = dict()
    YX = dict()
    for i in range(0, len(X)):
        x, y = X[i], Y[i]
        if x not in XY:
            XY[x] = y
        else:
            if XY[x] != y:
                return False
        if y not in YX:
            YX[y] = x
        else:
            if YX[y] != x:
                return False

    inv = {v: k for k, v in YX.items()}
    if inv != XY:
        return False
    return True


def matlab_connector(matlab_function, *args):
    matlab_code = "cd '{SHARED_MATLAB}'; ".format(SHARED_MATLAB=SHARED_MATLAB)
    matlab_code += 'format long; {matlab_function}({parameters}); exit'.format(matlab_function=matlab_function,
                                                                               parameters=",".join(
                                                                                   [str(a) for a in args]))
    command = [MATLAB_PATH, "-nodisplay", "-nosplash", "-nodesktop", "-r", matlab_code]
    result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode()
    # print(result)
    regexp = re.compile("\n(.*) =\n")
    split_list = regexp.split(result)[1:]
    result_dict = dict()
    for i in range(0, len(split_list), 2):
        name = split_list[i]
        value_string = split_list[i + 1].strip()
        value_array = np.array([])
        for line in value_string.split('\n'):
            line = line.strip()
            values = [float(x) for x in re.compile(' +').split(line)]
            line_array = np.array(values)
            value_array = np.vstack(
                (value_array if len(value_array) > 0 else np.empty(shape=(0, len(line_array))), line_array))
        result_dict[name] = value_array
    return result_dict


if __name__ == "__main__":
    data_file = "'/home/eremeykin/d_disk/projects/Clustering/ect/tests/shared/data/iris.pts'"
    data_file = "'/home/eremeykin/d_disk/projects/Clustering/ect/tests/shared/data/data500ws.pts'"
    threshold = 0
    p = 2
    beta = 2
    print(matlab_connector('test_ap_init_pb', data_file, threshold, p, beta))
