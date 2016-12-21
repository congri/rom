%oBFGS test script

%Define test function
%stochastic quadratic
dim = 4;
nSamples = 100;
J = eye(4);
samplesFun = @(theta, nS) normrnd(0, 1, dim, nS);
theta_opt = [1; 2; 3; 4];
meanGradFun = @(samples, theta) meanGrad(samples, theta, theta_opt, J);

%Setting params
RMstepWidth = 1;
RMoff = 100;
c = .1;
lambda = 1;
epsilon = 1e-10;
thetaInit = zeros(4, 1);



[optTheta] = oBFGS(meanGradFun, samplesFun, RMstepWidth, RMoff, c, lambda, epsilon, thetaInit, nSamples)



%% Function definitions
function [g] = meanGrad(samples, theta, theta_opt, J)
H = (1/size(samples, 2))*J*(samples*samples')*J';
g = H*(theta - theta_opt);
end