function [theta_c, theta_cf, domainc, domainf, phi, featureFunctionMean, featureFunctionSqMean,...
    featureFunctionMin, featureFunctionMax] = loadTrainedParams(datafolder)
%Load trained model parameters to workspace
%   Input:
%           datafolder:     folder of optimal training params

addpath('./params')
addpath('./featureFunctions')

%Load optimal params from job data
disp('Loading optimal parameters...')
theta_c.theta = dlmread(strcat('./data/', datafolder, '/theta'));
theta_c.theta = theta_c.theta(end, :)';
theta_c.Sigma = dlmread(strcat('./data/', datafolder, '/sigma'));
theta_c.Sigma = diag(theta_c.Sigma(end, :));
theta_cf.S = dlmread(strcat('./data/', datafolder, '/S'))';
W = dlmread(strcat('./data/', datafolder, '/Wmat'));
W = reshape(W, length(W)/3, 3)';
theta_cf.W = sparse(W(1, :), W(2, :), W(3, :));
theta_cf.mu = dlmread(strcat('./data/', datafolder, '/mu'))';
disp('done')

disp('Loading data normalization data...')
try
    featureFunctionMean = dlmread(strcat('./data/', datafolder, '/featureFunctionMean'));
    featureFunctionSqMean = dlmread(strcat('./data/', datafolder, '/featureFunctionSqMean'));
catch
    warning('featureFunctionMean, featureFunctionSqMean not found, setting it to 0.')
    featureFunctionMean = 0;
    featureFunctionSqMean = 0;
end

try
    featureFunctionMin = dlmread(strcat('./data/', datafolder, '/featureFunctionMin'));
    featureFunctionMax = dlmread(strcat('./data/', datafolder, '/featureFunctionMax'));
catch
    warning('featureFunctionMin, featureFunctionMax not found, setting it to 0.')
    featureFunctionMin = 0;
    featureFunctionMax = 0;
end
disp('done')

disp('Loading fine and coarse domain objects...')
loadTrainingData; %Contains domainf

if exist(strcat('./data/', datafolder, '/domainc.mat'), 'file')
    load(strcat('./data/', datafolder, '/domainc.mat'));
else
    warning('No coarse domain file found. Take boundary conditions from finescale data and regenerate. Please make sure everything is correct!')
    addpath('./heatFEM')
    jobname = datafolder;
    genCoarseDomain;
    rmpath('./heatFEM')
end
disp('done')

%Generate same basis functions as in training
disp('Setting up function handles to p_c basis functions...')
addpath('./rom')
genBasisFunctions;
disp('done')
rmpath('./rom')

end

