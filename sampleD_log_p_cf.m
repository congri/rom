%Script to sample d_log_p_cf under p_c

addpath('./heatFEM');

%Load finescale data
nStart = 1;
nTrain = 16;

mode = 'none';   %useDiagNeighbor, useLocalDiagNeigbor, useNeighbor, useLocalNeighbor, useLocal, linFiltSeq

loadTrainingData;
trainFilePath = ...
'~/matlab/data/fineData/systemSize=256x256/correlated_binary/IsoSEcov/l=20_sigmafSq=1/volumeFraction=0.2/locond=1_upcond=10/BCcoeffs=[0 1000 0 0]/set1-samples=1024.mat';
Tffile = matfile(trainFilePath);
Tf = Tffile.Tf(:, nStart:(nStart + nTrain - 1));        %Finescale temperatures - load partially to save memory

[theta_c, theta_cf, domainc, domainf, phi, featureFunctionMean, featureFunctionSqMean,...
    featureFunctionMin, featureFunctionMax] = loadTrainedParams('');

%comment this for inclusion of variances S of p_cf
theta_cf.S = ones(size(theta_cf.S));

theta_cf.Sinv = sparse(1:domainf.nNodes, 1:domainf.nNodes, 1./theta_cf.S);
theta_cf.Sinv_vec = 1./theta_cf.S;
%precomputation to save resources
theta_cf.WTSinv = theta_cf.W'*theta_cf.Sinv;

%% Compute design matrices
addpath('./rom');
Phi = DesignMatrix(domainf, domainc, phi, Tffile, nStart:(nStart + nTrain - 1));
load('./data/conductivityTransformation.mat');
%change here for anisotropy!
condTransOpts.anisotropy = false;
Phi = Phi.computeDesignMatrix(domainc.nEl, domainf.nEl, condTransOpts, mode);
%Normalize design matrices
%Phi = Phi.standardizeDesignMatrix(featureFunctionMean, featureFunctionSqMean);
% Phi = Phi.rescaleDesignMatrix(featureFunctionMin, featureFunctionMax);

nSamples = 1000;
d_log_p_cf_mean = 0;
d_log_p_cf_sqMean = 0;
k = 1;
% for i = nStart:(nStart + nTrain - 1)
%     for j = 1:nSamples
%         Xsample = mvnrnd(Phi.designMatrices{i}*theta_c.theta, theta_c.Sigma)';
%         conductivity = conductivityBackTransform(Xsample, condTransOpts);
%         [~, d_log_p_cf] = log_p_cf(Tf(:, i), domainc, conductivity, theta_cf, condTransOpts);
%         d_log_p_cf_mean = ((k - 1)/k)*d_log_p_cf_mean + (1/k)*d_log_p_cf;
%         d_log_p_cf_sqMean = ((k - 1)/k)*d_log_p_cf_sqMean + (1/k)*d_log_p_cf.^2;
%         k = k + 1;
%     end
% end

d_log_p_cf_mean
d_log_p_cf_var = d_log_p_cf_sqMean - d_log_p_cf_mean.^2
d_log_p_cf_std = sqrt(d_log_p_cf_var)
d_log_p_cf_err = d_log_p_cf_std/sqrt(nSamples*nTrain)
d_log_p_cf_sqMean
load('./data/noPriorSigma')
noPriorSigma
log_noPriorSigma = log(noPriorSigma)

disp('Sum of grad squares in x-direction:')
for i = 1:domainc.nElY
%     [i sum(d_log_p_cf_sqMean(((i - 1)*domainc.nElX + 1):(i*domainc.nElX)))]
%     log_noPriorSigma(((i - 1)*domainc.nElX + 1):(i*domainc.nElX))
    [i sum(log_noPriorSigma(((i - 1)*domainc.nElX + 1):(i*domainc.nElX)))]
    [i sum(noPriorSigma(((i - 1)*domainc.nElX + 1):(i*domainc.nElX)))]
end

disp('Sum of grad squares in y-direction:')
for i = 1:domainc.nElX
%     [i sum(d_log_p_cf_sqMean(i:domainc.nElX:((domainc.nElY - 1)*domainc.nElX + i)))]
%     log_noPriorSigma(i:domainc.nElX:((domainc.nElY - 1)*domainc.nElX + i))
    [i sum(log_noPriorSigma(i:domainc.nElX:((domainc.nElY - 1)*domainc.nElX + i)))]
    [i sum(noPriorSigma(i:domainc.nElX:((domainc.nElY - 1)*domainc.nElX + i)))]
end