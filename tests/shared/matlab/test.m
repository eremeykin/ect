data_file = '/home/eremeykin/d_disk/projects/Clustering/ect/tests/shared/data/iris.pts'
data = dlmread('/home/eremeykin/d_disk/projects/Clustering/ect/tests/shared/data/iris.pts');
ikThreshold = 0;
p = 2;
beta = 2;
test_ap_init_pb(data_file, ikThreshold, p, beta)
% f = Ward_pb_functions;
% [U, FinalW, InitW, FinalZ, InitZ, UDistToZ,LoopCount, AnomalousLabels] = f.iMWKmeans(data, ikThreshold, p, beta);
% Result = AnomalousLabels;
% Result
