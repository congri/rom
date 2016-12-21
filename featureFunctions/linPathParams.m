function [out] = linPathParams(cond, pathLengths, conductivities, domainc, domainf, param)
%Gives back the parameters a and b from the theoretical lineal path model
%L(z) = a*exp(b*z)

L = zeros(numel(pathLengths), 1);
nElc = [domainc.nElX domainc.nElY];
nElf = [domainf.nElX domainf.nElY];
for i = 1:numel(pathLengths)
    L(i) = .5*linealPath(cond, pathLengths(i), 'x', 1, conductivities, nElc, nElf)...
        + .5*linealPath(cond, pathLengths(i), 'y', 1, conductivities, nElc, nElf);
end

f = fit(pathLengths, L, 'exp1');
if strcmp(param, 'a')
    out = f.a;
elseif strcmp(param, 'b')
    out = f.b;
else
    error('Which Lineal path parameter?')
end

