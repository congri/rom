%%Script for output prediction
nSamples_p_c = 10000;
testSample_lo = 1;
testSample_up = 16;
testFilePath = '/home/constantin/matlab/data/fineData/systemSize=256x256/correlated_binary/IsoSEcov/l=5_sigmafSq=1/volumeFraction=0.3/locond=1_upcond=10/BCcoeffs=[-50 164 112 -30]/set2-samples=128.mat';
modelParamsFolder = '';

addpath('./aux')  %Is obsolete
addpath('./computation')
%execute prediction
[Tf_meanArray, Tf_varArray, Tf_mean_tot, Tf_sq_mean_tot, meanMahaErr, meanSqDist, sqDist, meanEffCond] =...
    predictOutput(nSamples_p_c, testSample_lo, testSample_up, testFilePath, modelParamsFolder);

%save predicted variables
save('./predictions.mat');
