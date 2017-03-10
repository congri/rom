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

%Catch infinities: Take maximum possible distance
if(any(any(isinf(dist))))
%     warning('Infinity in distance transformation. Setting to maximum possible distance')
    if strcmp(distMeasure, 'cityblock')
        dist(isinf(dist)) = size(dist, 1) + size(dist, 2);
    elseif strcmp(distMeasure, 'chessboard')
        dist(isinf(dist)) = max([size(dist, 1), size(dist, 2)]);
    else
        %Treat euclidean and quasi-euclidean equally. This is actually wrong for quasi-euclidean
        dist(isinf(dist)) = norm(size(dist));
    end
end

if(any(any(~isfinite(dist))))
    dist
    pause
end


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

