%Script to train binary autoencoder a la Tipping

close all;
clear;
addpath('./autoencoder')

%data specs
ba = BinaryAutoencoder;
ba.latentDim = 20;
ba.maxIterations = 10;
nElFX = 64;
nElFY = 64;
conductivityDistribution = 'correlated_binary';
volFrac = -1;
sigma_f2 = 1;
lengthscaleDist = 'lognormal';
lengthscaleParams = [-3 0.5];
loCond = 1;
upCond = 10;
boundaryConditions = '[0 1000 0 0]';
nSamples = 16384;
nStart = 1;
nTrain = 2048;

filename = strcat('~/matlab/data/fineData/systemSize=');
filename = strcat(filename, num2str(nElFX), 'x', num2str(nElFY), '/');
%Type of conductivity distribution
if strcmp(conductivityDistribution, 'correlated_binary')
    if strcmp(lengthscaleDist, 'delta')
        corrLength = lengthscaleParams(1);
        filename = strcat(filename, conductivityDistribution, '/', 'IsoSEcov/', 'l=',...
            num2str(corrLength), '_sigmafSq=', num2str(sigma_f2),...
            '/volumeFraction=', num2str(volFrac), '/', 'locond=',...
            num2str(loCond), '_upcond=', num2str(upCond),...
            '/', 'BCcoeffs=', boundaryConditions, '/');
    elseif strcmp(lengthscaleDist, 'lognormal')
        corrLengthMu = lengthscaleParams(1);
        corrLengthSigma = lengthscaleParams(2);
        filename = strcat(filename,...
            conductivityDistribution, '/', 'IsoSEcov/', 'l=lognormal_mu=',...
            num2str(corrLengthMu), 'sigma=', num2str(corrLengthSigma),...
            '_sigmafSq=', num2str(sigma_f2), '/volumeFraction=',...
            num2str(volFrac), '/', 'locond=', num2str(loCond),...
            '_upcond=', num2str(upCond),...
            '/', 'BCcoeffs=', boundaryConditions, '/');
    else
        error('Unknown length scale distribution')
    end
elseif strcmp(cond_distribution, 'binary')
    filename = strcat(filename,...
        conductivityDistribution, '/volumeFraction=',...
        num2str(volFrac), '/', 'locond=', num2str(loCond),...
        '_upcond=', num2str(upCond), '/', 'BCcoeffs=', boundaryConditions, '/');
else
    error('Unknown conductivity distribution')
end
%Name of training data file
filename = strcat(filename, 'set1-samples=', num2str(nSamples), '.mat');
matfile_cond = matfile(filename);


ba.trainingData = logical(matfile_cond.cond(:, nStart:(nStart + nTrain - 1)) - ...
    loCond);

ba = ba.train;
save('./autoencoder/trainedAutoencoder.mat', 'ba');




