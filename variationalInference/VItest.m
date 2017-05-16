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
VIparams.optParams.maxCompTime = 60;

minSamples = 100;
maxSamples = 200;
VIparams.optParams.nSamples = @(ii) nSamplesIteration(ii, minSamples, maxSamples);    %Gradient samples per iteration; given as a function with input iteration

if strcmp(VIparams.family, 'isotropicGaussian')
    %Take an isotropic Gaussian as the true distribution
    initialParams = [1 1];
    trueMu = [1 2 3 4];
    trueVar = 2;
    logTrueCondDist = @(x) -.5*VIparams.optParams.dim*log(2*pi) - .5*VIparams.optParams.dim*log(trueVar) - .5*(1/trueVar)*sum((x - trueMu).^2);
elseif strcmp(VIparams.family, 'diagonalGaussian')
    %Take a diagonal Gaussian as the true distribution
    initialParams = [-2*ones(1, VIparams.optParams.dim) -1.38*ones(1, VIparams.optParams.dim)];
    trueMu = [1 2 -3];
    truePrecision = inv([3 0 1; 0 5 0; 1 0 2]);
%     trueCovarInvDiagMat = sparse(1:dim, 1:dim, 1./trueCovarDiag);
%     logTrueCondDist = @(x) -.5*dim*log(2*pi) - .5*sum(log(trueCovarDiag)) - .5*(x - trueMu)*trueCovarInvDiagMat*(x - trueMu)';
    logTrueCondDist = @(x) testGaussian(x, truePrecision, trueMu);
elseif strcmp(VIparams.family, 'fullRankGaussian')
    %Take a diagonal Gaussian as the true distribution
    initialParams = 100*eye(2);
    trueMu = [1 2];
    trueCovarDiag = [16 4];
    trueCovarInvDiagMat = sparse(1:VIparams.optParams.dim, 1:VIparams.optParams.dim, 1./trueCovarDiag);
    logTrueCondDist = @(x) -.5*VIparams.optParams.dim*log(2*pi) - .5*sum(log(trueCovarDiag)) - .5*(x - trueMu)*trueCovarInvDiagMat*(x - trueMu)';

end

% [optVarDist, RMsteps] = variationalInference(logTrueCondDist, VIparams, initialParams)
% finalParams = [optVarDist.params(1:(numel(optVarDist.params)/2))...
%     exp(-optVarDist.params(((numel(optVarDist.params)/2)) + 1:end))]





testDiag = false;
if testDiag
    %Test new VI and stochastic optimization class
    so = StochasticOptimization('robbinsMonro');
    so.stepWidth = 2e-1;
    varDistParams.mu = [0, 0];
    varDistParams.sigma = [1, 1];
    so.x = [varDistParams.mu, -2*log(varDistParams.sigma)];
    
    ELBOgradParams.nSamples = 100;
    vi = VariationalInference(logTrueCondDist, 'diagonalGauss', varDistParams, ELBOgradParams);
    
    
    for i = 1:10000
        [ELBOgrad, ELBOgradErr] = vi.sampleELBOgrad;
        
        so.gradient = ELBOgrad;
        so = so.update;
        new_mu = so.x(1:vi.dim)
        new_sigma = exp(-.5*so.x((vi.dim + 1):end))
        params.mu = new_mu;
        params.sigma = new_sigma;
        vi = vi.setVarDistParams(params);
    end
end


testFull = true;
if testFull
    %Test new VI and stochastic optimization class
    so = StochasticOptimization('adam');
    so.stepWidth = 1e-2;
    varDistParams.mu = [0, 0, 0];
    varDistParams.Sigma = 1e0*eye(length(varDistParams.mu));
    varDistParams.LT = chol(varDistParams.Sigma);
    varDistParams.LMinusT = inv(varDistParams.LT);
    varDistParams.L = varDistParams.LT';
    varDistParams.LInv = inv(varDistParams.L);
    so.x = [varDistParams.mu, varDistParams.L(:)'];
    
    ELBOgradParams.nSamples = 10;
    vi = VariationalInference(logTrueCondDist, 'fullRankGauss', varDistParams, ELBOgradParams);
    i = 0;
    while true
        [ELBOgrad, ELBOgradErr] = vi.sampleELBOgrad;
        so.gradient = ELBOgrad;
        so = so.update;
        new_mu = so.x(1:vi.dim);
        new_L = so.x((vi.dim + 1):end)';
        new_L = reshape(new_L, vi.dim, vi.dim);
        d_L = reshape(ELBOgrad((vi.dim + 1):end)', vi.dim, vi.dim);
        new_Sigma = new_L*new_L';
        params.mu = new_mu;
        params.Sigma = new_Sigma;
        params.LT = new_L';
        params.L = new_L;
        params.LInv = inv(params.L);
        vi = vi.setVarDistParams(params);
        i = i + 1;
        
        if(mod(i, 1000) == 0)
            new_mu
            new_Sigma
        end
    end
end







