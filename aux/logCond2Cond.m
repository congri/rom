function [lambda] = logCond2Cond(X, lowerBound, upperBound)
%Converts log conductivity field X to conductivity field lambda

lambda = exp(X);
if any(lambda < lowerBound)
    %lower bound on conductivity for stability
    lambda(lambda < lowerBound) = lowerBound;
    warning('Effective conductivity is below lower bound')
end
if any(lambda > upperBound)
    %upper bound on conductivity for stability
    lambda(lambda > upperBound) = upperBound;
    warning('Effective conductivity is above upper bound')
end

end

