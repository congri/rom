function [out] = distanceProps(lambdaMat, conductivities, hilo, distMeasure, meanVarMaxMin)
%Uses built-in Matlab 'bwdist' to return mean/var/max/min of pixel distances to next phase
%   lambdaMat:          2-dim conductivity image
%   conductivities:     loCond in first, upCond in second entry
%   hilo:               property for high or low phase bubbles?
%   distMeasure: 'euclidean', 'chessboard', 'cityblock', 'quasi-euclidean'
%See matlab reference for bwdist


%Convert lambda to binary image
if strcmp(hilo, 'hi')
    lambdaMat = (lambdaMat > conductivities(1));
elseif strcmp(hilo, 'lo')
    lambdaMat = (lambdaMat < conductivities(2));
else
    error('Property of high or low conducting phase?')
end

dist = bwdist(lambdaMat, distMeasure);


m = mean(mean(dist));
Max = max(max(dist));
Min = min(min(dist));
v = var(dist(:));

if strcmp(meanVarMaxMin, 'mean')
    out = m;
elseif strcmp(meanVarMaxMin, 'var')
    out = v;
elseif strcmp(meanVarMaxMin, 'max')
    out = Max;
elseif strcmp(meanVarMaxMin, 'min')
    warning('Minimum distance is usually 0 for every macro element. This feature should not be used.')
    out = Min;
else
    error('Mean or variance of lambda bubble property?')
end


end

