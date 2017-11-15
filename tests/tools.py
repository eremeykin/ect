
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

