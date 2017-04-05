%%Script for output prediction
nSamples_p_c = 1000;
testSample_lo = 1;
testSample_up = 128;
testFilePath = ...
'~/matlab/data/fineData/systemSize=256x256/correlated_binary/IsoSEcov/l=20_sigmafSq=1/volumeFraction=0.2/locond=1_upcond=10/BCcoeffs=[0 1000 0 0]/set1-samples=1024.mat';
modelParamsFolder = '';
mode = 'useLocal';   %useDiagNeighbor, useNeighbor, useLocal

addpath('./aux')  %Is obsolete
addpath('./computation')
%execute prediction
[Tf_meanArray, Tf_varArray, Tf_mean_tot, Tf_sq_mean_tot, meanMahaErr, meanSqDist, sqDist, meanEffCond, meanSqDistErr] =...
    predictOutput(nSamples_p_c, testSample_lo, testSample_up, testFilePath, modelParamsFolder, mode, theta_c);

save('./predictions.mat', 'Tf_meanArray', 'Tf_varArray', 'Tf_sq_mean_tot',...
    'meanMahaErr', 'meanSqDist', 'sqDist', 'meanEffCond', 'meanSqDistErr');
