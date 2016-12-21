function [maxExt] = maxExtent(lambdaMat, conductivities, hilo, dir)
%Uses built-in Matlab 'regionprops' 'Extrema' to calculate maximum extent in x and y directions
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

maxExt = 0;
for i = 1:numel(extrema)
    blob_extrema = extrema{i};
    if strcmp(dir, 'x')
        if((blob_extrema(3, 1) - blob_extrema(8, 1)) > maxExt)
            maxExt = blob_extrema(3, 1) - blob_extrema(8, 1);
        end
    elseif strcmp(dir, 'y')
        if((blob_extrema(6, 2) - blob_extrema(1, 2)) > maxExt)
            maxExt = blob_extrema(6, 2) - blob_extrema(1, 2);
        end
    end
end
assert(maxExt >= 0, 'Error: maximum blob extention < 0');


end

