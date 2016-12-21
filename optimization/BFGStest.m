%Test script for BFGS maximization
startValue = [0; 0];
options.Xtol = 1e-6;
options.initialStepSize = 1;
options.debug = false;

tfunHandle = @(x) tfun(x);
options.provide_objective = false;
[curr_x, grad, nIter] = BFGSMaximization(tfunHandle, startValue, options)

%Test function with maximum at [4; -.5]
function [grad] = tfun(x)
    grad = [-2*x(1) + 8; -2*x(2) - 1];
end