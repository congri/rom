function [out] = generalizedMeanPath(lambdak, dir, meanParam, mode)
%Generalized mean on a straight path from left to right/ top to bottom
%   lambdak:        fine conductivities in coarse element k
%   dir:            x or y direction
%   phase:          Material phase, 1 or 2 for binary
%   meanParam:      Generalized mean parameter
%
%   Output:
%       N:          Number of pixels of phase phase that have to be crossed going from one side to
%                   another. mean/max/min/var is possible, whereas min and max should be
%                   complementary with phase exchange

%Check
% assert(strcmp(fineData.dist, 'binary'), 'Error: linealPath is only a possible basis function if conductivities are binary')

if dir == 'x'
    m = zeros(1, size(lambdak, 2));
    for i = 1:size(lambdak, 2)
        m(i) = generalizedMean(lambdak(:, i), meanParam);
    end
elseif dir == 'y'
    m = zeros(1, size(lambdak, 1));
    for i = 1:size(lambdak, 1)
        m(i) = generalizedMean(lambdak(i, :), meanParam);
    end
else
    error('Unknown direction for generalizedMeanPath function')
end

if strcmp(mode, 'mean')
    out = mean(m);
elseif strcmp(mode, 'max')
    out = max(m);
elseif strcmp(mode, 'min')
    out = min(m);
elseif strcmp(mode, 'var')
    out = var(m);
else
    error('Unknown mode')
end


end