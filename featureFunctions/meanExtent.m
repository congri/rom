function [out] = meanExtent(lambdaMat, conductivities, hilo, dir, meanOrVar)
%Uses built-in Matlab 'regionprops' 'Extrema' to calculate mean extent in x and y directions
%   lambdaMat:     2-dim conductivity image
%   fineData:   fineData structure containing high and low phase conductivity
%   hilo:       property for high or low phase bubbles?


%Convert lambda to binary image
if strcmp(hilo, 'hi')
    lambdaMat = (lambdaMat > conductivities(1));
elseif strcmp(hilo, 'lo')
    lambdaMat = (lambdaMat < conductivities(2));
else
    error('Property of high or low conducting phase?')
end

extrema = regionprops(lambdaMat, 'Extrema');
extrema = struct2cell(extrema);

meanExt = 0;
meanExtSq = 0;
for i = 1:numel(extrema)
    blob_extrema = extrema{i};
    if strcmp(dir, 'x')
        maxExt_blob = blob_extrema(3, 1) - blob_extrema(8, 1);
        meanExt = meanExt + maxExt_blob;
        meanExtSq = meanExtSq + maxExt_blob.^2;
    elseif strcmp(dir, 'y')
        maxExt_blob = blob_extrema(6, 2) - blob_extrema(1, 2);
        meanExt = meanExt + maxExt_blob;
        meanExtSq = meanExtSq + maxExt_blob.^2;
    end
end
meanExt = meanExt/numel(extrema);
meanExtSq = meanExtSq/numel(extrema);

if strcmp(meanOrVar, 'mean')
    out = meanExt;
elseif strcmp(meanOrVar, 'var')
    out = meanExtSq - meanExt^2;
else
    error('Mean or variance?')
end

end