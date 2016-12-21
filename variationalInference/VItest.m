%Test script to test variational inference

clear all
%Set VI params
dim = 1;        %ATTENTION: set params accordingly
params.family = 'diagonalGaussian';
params.RMtype = 'adam';
params.initialParams{1} = zeros(1, dim);
params.nSamples = 100;
params.robbinsMonro.stepWidth = 70;
params.robbinsMonro.offset = 1000;
params.robbinsMonro.relXtol = 1e-8;
params.decayParam = .9;
params.adam.beta1 = .9;
params.adam.beta2 = .999;

if strcmp(params.family, 'isotropicGaussian')
    %Take an isotropic Gaussian as the true distribution
    params.initialParams{2} = 1;
    trueMu = [1 2 3 4];
    trueVar = 2;
    dim = length(trueMu);
    logTrueCondDist = @(x) -.5*dim*log(2*pi) - .5*dim*log(trueVar) - .5*(1/trueVar)*sum((x - trueMu).^2);
elseif strcmp(params.family, 'diagonalGaussian')
    %Take a diagonal Gaussian as the true distribution
    params.initialParams{2} = 4*ones(1, dim);
    trueMu = [1];
    trueCovarDiag = [4];
    dim = length(trueMu);
%     trueCovarInvDiagMat = sparse(1:dim, 1:dim, 1./trueCovarDiag);
%     logTrueCondDist = @(x) -.5*dim*log(2*pi) - .5*sum(log(trueCovarDiag)) - .5*(x - trueMu)*trueCovarInvDiagMat*(x - trueMu)';
    logTrueCondDist = @(x) testGaussian(x, trueCovarDiag, trueMu);
elseif strcmp(params.family, 'fullRankGaussian')
    %Take a diagonal Gaussian as the true distribution
    params.initialParams{2} = 100*eye(2);
    trueMu = [1 2];
    trueCovarDiag = [16 4];
    dim = length(trueMu);
    trueCovarInvDiagMat = sparse(1:dim, 1:dim, 1./trueCovarDiag);
    logTrueCondDist = @(x) -.5*dim*log(2*pi) - .5*sum(log(trueCovarDiag)) - .5*(x - trueMu)*trueCovarInvDiagMat*(x - trueMu)';

end

[optVarDist, RMsteps] = varInf(logTrueCondDist, params)