function [L] = linealPath(lambdak, pathLength, dir, phase, phaseConductivity, nElc, nElf)
%Lineal path function to use as a basis function phi in p_c
%   lambdak:        fine conductivities in coarse element k
%   pathLength:     length of path in units of fine elements. Must be smaller than sqrt(nFine/nCoarse)
%   dir:            x or y direction
%   phase:          Material phase, 1 or 2 for binary
%   fineData:       distribution type of fine conductivities. Must be binary!
%
%   Output:
%       L:          Lineal path function to given parameters, see Torquato p. 44

%Check
% assert(strcmp(fineData.dist, 'binary'), 'Error: linealPath is only a possible basis function if conductivities are binary')

lambdakBin = (lambdak == phaseConductivity(phase));

%Fine elements per coarse element in x and y directions
xc = nElf(1)/nElc(1);
yc = nElf(2)/nElc(2);

if dir == 'x'
    %Maximal number of line segments of length pathLength that can be dropped in a single row/column in
    %x/y direction
    maxX = xc - pathLength;
    assert(maxX >= 1, 'Error: maximal number of possible line segments per row must be at least 1')
    samePhase = 0;
    i = 1;
    for row = 1:yc
        for col = 1:maxX
            if(all(lambdakBin(i:(i + pathLength))))
                samePhase = samePhase + 1;
            end
            i = i + 1;
        end
        i = i + pathLength;   %jump over to first column in next row
    end
    L = samePhase/(maxX*yc);
    
elseif dir == 'y'
    %Maximal number of line segments of length pathLength that can be dropped in a single row/column in
    %x/y direction
    maxY = yc - pathLength;
    assert(maxY >= 1, 'Error: maximal number of possible line segments per column must be at least 1')
    samePhase = 0;
    i = 1;
    for row = 1:maxY
        for col = 1:xc
            if(all(lambdakBin(i:xc:(i + pathLength*xc))))
               samePhase = samePhase + 1; 
            end
            i = i + 1;
        end
        %no jump needed here?
    end
    L = samePhase/(maxY*xc);
    
else
    error('Unknown direction for linealPath function')
end


end

