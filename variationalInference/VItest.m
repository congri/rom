%Test script to test variational inference

clear all


%Take a diagonal Gaussian as the true distribution
trueMu = [1 2 -3];
truePrecision = inv([3 0 1; 0 5 0; 1 0 2]);
logTrueCondDist = @(x) testGaussian(x, truePrecision, trueMu);


testDiag = false;
if testDiag
    %Test new VI and stochastic optimization class
    so = StochasticOptimization('adam');
    so.stepWidth = 2e-1;
    varDistParams.mu = [0, 0, 0];
    varDistParams.sigma = [1, 1, 1];
    so.x = [varDistParams.mu, -2*log(varDistParams.sigma)];
    
    ELBOgradParams.nSamples = 100;
    vi = VariationalInference(logTrueCondDist, 'diagonalGauss', varDistParams, ELBOgradParams);
    so.gradientHandle = @(x) vi.gradientHandle(x);
    so = so.converge;
    
    vi = vi.setVarDistParams(vi.params_vec2struc(so.x));
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
    so.gradientHandle = @(x) vi.gradientHandle(x);
    so = so.converge;
    
    vi = vi.setVarDistParams(vi.params_vec2struc(so.x));
end







