function [N] = nPixelCross(lambdak, dir, phase, phaseConductivity, mode)
%Number of pixels with phase phase going from left to right/ top to bottom on a straight line
%   lambdak:        fine conductivities in coarse element k
%   dir:            x or y direction
%   phase:          Material phase, 1 or 2 for binary
%
%   Output:
%       N:          Number of pixels of phase phase that have to be crossed going from one side to
%                   another. mean/max/min/var is possible, whereas min and max should be
%                   complementary with phase exchange

%Check
% assert(strcmp(fineData.dist, 'binary'), 'Error: linealPath is only a possible basis function if conductivities are binary')

%Fine elements per coarse element in x and y directions
xc = size(lambdak, 1);
yc = size(lambdak, 2);

lambdakBin = (lambdak == phaseConductivity(phase));

if dir == 'x'
    N_vec = sum(lambdakBin)/xc;
    
elseif dir == 'y'
    N_vec = sum(lambdakBin, 2)/yc;
    
else
    error('Unknown direction for nPixelCross function')
end

if strcmp(mode, 'mean')
    N = mean(N_vec);
elseif strcmp(mode, 'max')
    N = max(N_vec);
elseif strcmp(mode, 'min')
    N = min(N_vec);
elseif strcmp(mode, 'var')
    N = var(N_vec);
else
    error('Unknown mode')
end


end


