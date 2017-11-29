function [ Result ] = test_imwk_means_pb(data_file, ikThreshold, p, beta)
data = dlmread(data_file);
f = Ward_pb_functions;
[U, FinalW, InitW, FinalZ, InitZ, UDistToZ,LoopCount, AnomalousLabels] = f.iMWKmeans(data, ikThreshold, p, beta);
Result = U;
Result
end
