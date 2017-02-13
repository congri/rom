function [m] = generalizedMeanBoundary(lambda, nElc, nElf, meanParam, boundary)
%Computes the mean of lambda along a specified boundary

%Fine elements per coarse element in x and y directions
xc = nElf(1)/nElc(1);
yc = nElf(2)/nElc(2);

lambda = reshape(lambda, xc, yc);

if strcmp(boundary, 'left')
    m = generalizedMean(lambda(:, 1), meanParam);
elseif strcmp(boundary, 'lower')
    m = generalizedMean(lambda(end, :), meanParam);
elseif strcmp(boundary, 'right')
    m = generalizedMean(lambda(:, end), meanParam);
elseif strcmp(boundary, 'upper')
    m = generalizedMean(lambda(1, :), meanParam);
else
    error('Unknown boundary')
end


end

