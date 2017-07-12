%Script to train binary autoencoder a la Tipping

close all;
clear;
addpath('./autoencoder')

%data specs
ba = BinaryAutoencoder;
ba.latentDim = 20;
ba.maxIterations = 10;
nElFX = 256;
nElFY = 256;
conductivityDistribution = 'correlated_binary';
volFrac = -1;
sigma_f2 = 1;
lengthscaleDist = 'lognormal';
lengthscaleParams = [-3 0.5];
loCond = 1;
upCond = 10;
boundaryConditions = '[0 1000 0 0]';
nSamples = 1024;
nStart = 1;
nTrain = 1024;  %for autoencoder

folder = strcat('~/matlab/data/fineData/systemSize=');
folder = strcat(folder, num2str(nElFX), 'x', num2str(nElFY), '/');
%Type of conductivity distribution
if strcmp(conductivityDistribution, 'correlated_binary')
    if strcmp(lengthscaleDist, 'delta')
        corrLength = lengthscaleParams(1);
        folder = strcat(folder, conductivityDistribution, '/', 'IsoSEcov/', 'l=',...
            num2str(corrLength), '_sigmafSq=', num2str(sigma_f2),...
            '/volumeFraction=', num2str(volFrac), '/', 'locond=',...
            num2str(loCond), '_upcond=', num2str(upCond),...
            '/', 'BCcoeffs=', boundaryConditions, '/');
    elseif strcmp(lengthscaleDist, 'lognormal')
        corrLengthMu = lengthscaleParams(1);
        corrLengthSigma = lengthscaleParams(2);
        folder = strcat(folder,...
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
    folder = strcat(folder,...
        conductivityDistribution, '/volumeFraction=',...
        num2str(volFrac), '/', 'locond=', num2str(loCond),...
        '_upcond=', num2str(upCond), '/', 'BCcoeffs=', boundaryConditions, '/');
else
    error('Unknown conductivity distribution')
end
%Name of training data file
trainingDataFilename = strcat(folder, 'set1-samples=', num2str(nSamples), '.mat');
matfile_cond = matfile(trainingDataFilename);
cond = matfile_cond.cond(:, nStart:(nStart + nTrain - 1));


ro = ROM_SPDE;
try
    load(strcat(folder, 'fineScaleDomain.mat'));
catch
    temp = load(strcat(folder, 'romObj.mat'));
end
[lambdak] = getCoarseElementConductivity(ro.coarseScaleDomain,...
    fineScaleDomain, cond);

plotCoarseWindows = false;
if plotCoarseWindows
    f1 = figure;
    f2 = figure;
    window = 1:ro.coarseScaleDomain.nEl;
    window = reshape(window, ro.coarseScaleDomain.nElX, ro.coarseScaleDomain.nElY);
    window = window';
    window = window(:);
    for n = 1:nTrain
        figure(f1);
        imagesc(reshape(cond(:, n), fineScaleDomain.nElX, fineScaleDomain.nElY));
        axis square;
        grid off;
        xticks({});
        yticks({});
        
        figure(f2)
        for k = 1:ro.coarseScaleDomain.nEl
            subplot(ro.coarseScaleDomain.nElX, ro.coarseScaleDomain.nElY, window(k))
            imagesc(lambdak{n, k})
            axis square;
            grid off;
            xticks({});
            yticks({});
        end
        drawnow;
        pause
    end
end
lambdakMat = zeros(numel(lambdak{1}), numel(lambdak));
i = 1;
for n = 1:size(lambdak, 1)
    for k = 1:size(lambdak, 2)
    lambdakMat(:, i) = lambdak{n, k}(:);
    i = i + 1;
    end
end

ba.trainingData = logical(lambdakMat - loCond);

ba = ba.train;
save('./autoencoder/trainedAutoencoder.mat', 'ba');




