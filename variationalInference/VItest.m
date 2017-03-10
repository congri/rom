%Test script to test variational inference

clear all
%Set VI params
VIparams.family = 'diagonalGaussian';
VIparams.nSamples = 10000;    %Gradient samples per iteration
VIparams.inferenceSamples = 200;
VIparams.optParams.optType = 'adam';
VIparams.optParams.dim = 2;
VIparams.optParams.stepWidth = .1;
VIparams.optParams.XWindow = 20;    %Averages dependent variable over last iterations
VIparams.optParams.offset = 10000;  %Robbins-Monro offset
VIparams.optParams.relXtol = 1e-12;
VIparams.optParams.maxIterations = 500;
VIparams.optParams.meanGradNormTol = .001;    %Converged if norm of mean of grad over last k iterations is smaller
VIparams.optParams.gradNormTol = .001;    %Converged if average norm of gradient in last gradNormWindow iterations is below
VIparams.optParams.gradNormWindow = 30;  %gradNormTol
VIparams.optParams.decayParam = .7;   %only works for diagonal Gaussian
VIparams.optParams.adam.beta1 = .9;     %The higher this parameter, the more gradient information from previous steps is retained
VIparams.optParams.adam.beta2 = .999;

if strcmp(VIparams.family, 'isotropicGaussian')
    %Take an isotropic Gaussian as the true distribution
    initialParams = [1 1];
    trueMu = [1 2 3 4];
    trueVar = 2;
    logTrueCondDist = @(x) -.5*VIparams.optParams.dim*log(2*pi) - .5*VIparams.optParams.dim*log(trueVar) - .5*(1/trueVar)*sum((x - trueMu).^2);
elseif strcmp(VIparams.family, 'diagonalGaussian')
    %Take a diagonal Gaussian as the true distribution
    initialParams = [-2*ones(1, VIparams.optParams.dim) -1.38*ones(1, VIparams.optParams.dim)];
    trueMu = [1 4];
    trueCovarDiag = [1 5];
%     trueCovarInvDiagMat = sparse(1:dim, 1:dim, 1./trueCovarDiag);
%     logTrueCondDist = @(x) -.5*dim*log(2*pi) - .5*sum(log(trueCovarDiag)) - .5*(x - trueMu)*trueCovarInvDiagMat*(x - trueMu)';
    logTrueCondDist = @(x) testGaussian(x, trueCovarDiag, trueMu);
elseif strcmp(VIparams.family, 'fullRankGaussian')
    %Take a diagonal Gaussian as the true distribution
    initialParams = 100*eye(2);
    trueMu = [1 2];
    trueCovarDiag = [16 4];
    trueCovarInvDiagMat = sparse(1:VIparams.optParams.dim, 1:VIparams.optParams.dim, 1./trueCovarDiag);
    logTrueCondDist = @(x) -.5*VIparams.optParams.dim*log(2*pi) - .5*sum(log(trueCovarDiag)) - .5*(x - trueMu)*trueCovarInvDiagMat*(x - trueMu)';

end

[optVarDist, RMsteps] = variationalInference(logTrueCondDist, VIparams, initialParams)