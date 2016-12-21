function [N, E] = shapeFunctionValues(domain, x)
%coarse element
row = floor(x(2)/domain.lElY) + 1;
%upper boundary of domain
if row > domain.nElY
    row = domain.nElY;
end
col = floor(x(1)/domain.lElX) + 1;
%right boundary of domain
if col > domain.nElX
    col = domain.nElX;
end
%E is coarse element x is in
E = (row - 1)*(domain.nElX) + col;

%shape function values
N(1) =(1/domain.AEl)*(x(1) - domain.lc(E, 2, 1))*(x(2) - domain.lc(E, 4, 2));
N(2,1) = -(1/domain.AEl)*(x(1) - domain.lc(E, 1, 1))*(x(2) - domain.lc(E, 4, 2));
N(3) = (1/domain.AEl)*(x(1) - domain.lc(E, 1, 1))*(x(2) - domain.lc(E, 1, 2));
N(4) = -(1/domain.AEl)*(x(1) - domain.lc(E, 2, 1))*(x(2) - domain.lc(E, 1, 2));
end