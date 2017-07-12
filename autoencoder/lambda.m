function [l] = lambda(xi)
%l = (.5 - sigmoid(xi))./(2*xi + eps);
l = (.5 - 1./(1 + exp(-xi)))./(2*xi);
l(xi == 0) = -.125;
end

