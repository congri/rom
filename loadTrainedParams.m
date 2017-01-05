function [theta_c, theta_cf, domainc, domainf, phi, featureFunctionAbsMean] = loadTrainedParams(datafolder)
%Load trained model parameters to workspace
%   Input:
%           datafolder:     folder of optimal training params

addpath('./params')
addpath('./featureFunctions')

%Load optimal params from job data
disp('Loading optimal parameters...')
theta_c.theta = dlmread(strcat('./data/', datafolder, '/theta'));
theta_c.theta = theta_c.theta(end, :)';
theta_c.sigma = dlmread(strcat('./data/', datafolder, '/sigma'));
theta_c.sigma = theta_c.sigma(end);
theta_cf.S = dlmread(strcat('./data/', datafolder, '/S'))';
W = dlmread(strcat('./data/', datafolder, '/Wmat'));
theta_cf.W = sparse(W(1, :), W(2, :), W(3, :));
theta_cf.mu = dlmread(strcat('./data/', datafolder, '/mu'))';
disp('done')

disp('Loading column norms of Phi of training data...')
featureFunctionAbsMean = dlmread(strcat('./data/', datafolder, '/featureFunctionAbsMean'));
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

