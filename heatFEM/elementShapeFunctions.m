function [N] = elementShapeFunctions(x, y, xe, Ael)
%Gives values of element shape functions
%   x, y:  domain variables
%   xe: xe(1) = x_1^e, xe(2) = x_2^e, xe(3) = y_1^e, xe(4) = y_4^e, see Fish&Belytschko p163

N = zeros(4, 1);
N(1) = (x - xe(2))*(y - xe(4));
N(2) = -(x - xe(1))*(y - xe(4));
N(3) = (x - xe(1))*(y - xe(3));
N(4) = -(x - xe(2))*(y - xe(3));
N = (1/Ael)*N;

end

