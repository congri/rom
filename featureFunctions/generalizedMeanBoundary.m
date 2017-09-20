function [m] = generalizedMeanBoundary(lambda, meanParam, boundary)
%Computes the mean of lambda along a specified boundary

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

