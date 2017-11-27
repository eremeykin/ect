data = dlmread('/home/eremeykin/d_disk/projects/Clustering/ect/tests/api_p_beta/data/symmetric_15points.pts');
ikThreshold = 0;
p = 2;
beta = 2;
f = Ward_pb_functions;
f.iMWKmeans(data, ikThreshold, p, beta)

