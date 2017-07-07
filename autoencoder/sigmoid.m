function [s] = sigmoid(x, x_0, steepness, maximum)
%Sigmoid function

if nargin < 2
    x_0 = 0;
end
if nargin < 3
    steepness = 1;
end
if nargin < 4
    maximum = 1;
end

s = maximum./(1 + exp(-steepness.*(x - x_0)));

end

