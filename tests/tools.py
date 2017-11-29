MATLAB_PATH = "/home/eremeykin/Matlab2015b/bin/matlab"
SHARED_MATLAB = "/media/d_disk/projects/Clustering/ect/tests/shared/matlab"
import subprocess


def transformation_exists(X, Y):
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
    matlab_code += '{matlab_function}({parameters}); exit'.format(matlab_function=matlab_function,
                                                            parameters=",".join([str(a) for a in args]))
    command = [MATLAB_PATH, "-nodisplay", "-nosplash", "-nodesktop", "-r", matlab_code]
    result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode()
    end = result.find("=")
    return result[end + 2:].split()


if __name__ == "__main__":
    data_file = "'/home/eremeykin/d_disk/projects/Clustering/ect/tests/shared/data/iris.pts'"
    threshold = 0
    p = 2
    beta = 2
    matlab_connector('test_ap_init_pb', data_file, threshold, p, beta)

