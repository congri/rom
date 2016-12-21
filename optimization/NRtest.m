%Test script for Newton-Raphson maximization
startValue = [0; 0];
Xtol = 1e-6;

tfunHandle = @(x) tfun(x);
provide_objective = false;
[curr_x, grad, Hess, nIter] = newtonRaphsonMaximization(tfunHandle, startValue, Xtol, provide_objective)

%Test function
function [grad, Hess, obj] = tfun(x)
    obj = -(x(1) - 3)^2*(3*x(2) + 1)^2;
    grad = [-2*(x(1) - 3)*(3*x(2) + 1)^2; -6*(3*x(2) + 1)*(x(1) - 3)^2];
    offDiag = - 12*(3*x(2) + 1)*(x(1) - 3);
    Hess = [-2*(3*x(2) + 1)^2, offDiag; offDiag, -18*(x(1) - 3)^2];
end
