%%Script for output prediction
nSamples_p_c = 1000;
testSample_lo = 1;
testSample_up = 128;
testFilePath = ...
'~/matlab/data/fineData/systemSize=256x256/correlated_binary/IsoSEcov/l=20_sigmafSq=1/volumeFraction=0.08/locond=1_upcond=10/BCcoeffs=[-50 164 112 -30]/set2-samples=256.mat';
modelParamsFolder = '';
useNeighbor = false; %include feature information from neighboring coarse elements?
useLocal = false;     %use separate theta_c for each macro-cell?
assert(~(useNeighbor && useLocal), 'useNeighbor and useLocal cannot be used at the same time')

addpath('./aux')  %Is obsolete
addpath('./computation')
%execute prediction
[Tf_meanArray, Tf_varArray, Tf_mean_tot, Tf_sq_mean_tot, meanMahaErr, meanSqDist, sqDist, meanEffCond, meanSqDistErr] =...
    predictOutput(nSamples_p_c, testSample_lo, testSample_up, testFilePath, modelParamsFolder, useNeighbor, useLocal);

save('./predictions.mat', 'Tf_meanArray', 'Tf_varArray', 'Tf_sq_mean_tot',...
    'meanMahaErr', 'meanSqDist', 'sqDist', 'meanEffCond', 'meanSqDistErr');
