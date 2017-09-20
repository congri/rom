function [out] = linPathParams(cond, pathLengths, conductivities, phase, param)
%Gives back the parameters a and b from the theoretical lineal path model
%L(z) = a*exp(b*z)

L = zeros(numel(pathLengths), 1);
for i = 1:numel(pathLengths)
    L(i) = .5*linealPath(cond, pathLengths(i), 'x', phase, conductivities)...
        + .5*linealPath(cond, pathLengths(i), 'y', phase, conductivities);
end

L = L + eps;
f = fit(pathLengths, L, 'exp1');
if strcmp(param, 'a')
    out = f.a;
elseif strcmp(param, 'b')
    out = f.b;
else
    error('Which Lineal path parameter?')
end

