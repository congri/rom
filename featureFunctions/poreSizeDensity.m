function [p_delta_x, p_delta] = poreSizeDensity(lambda, phase, phaseConductivity, nElc, nElf)
%Pore-size density function of materials
%   lambdak:        fine conductivities in coarse element k
%   distance:     length of distance in units of fine elements (pixels). Must be smaller than sqrt(nFine/nCoarse)
%   dir:            x or y direction
%   phase:          Material phase, 1 or 2 for binary
%   fineData:       distribution type of fine conductivities. Must be binary!
%
%   Output:
%       p_delta:          pore size density function to given parameters, see Torquato p. 48
%       p_delta_x:        x-coordinate (random variable) of pore size density

%Fine elements per coarse element in x and y directions
xc = nElf(1)/nElc(1);
yc = nElf(2)/nElc(2);

p_delta = zeros(max([xc yc]) + 1, 1);
p_delta_x = (0:max([xc yc]))';

i = 1;
for row = 1:yc
    for col = 1:xc
        if(lambda(i) == phaseConductivity(phase))
            j = 0;
            %left side
            while(j < col)
                if(lambda(i - j) == phaseConductivity(phase))
                    delta_l = j;
                end
                j = j + 1;
            end
            
            j = 0;
            %right side
            while(j + col <= xc)
                if(lambda(i + j) == phaseConductivity(phase))
                    delta_r = j;
                end
                j = j + 1;
            end
            
            j = 0;
            %upper side
            while(row + j <= yc)
                if(lambda(i + j*xc) == phaseConductivity(phase))
                    delta_u = j;
                end
                j = j + 1;
            end
            
            j = 0;
            %lower side
            while(j < row)
                if(lambda(i - j*xc) == phaseConductivity(phase))
                    delta_d = j;
                end
                j = j + 1;
            end
            
            delta = min([delta_l delta_r delta_u delta_d]);
            p_delta(delta + 1) = p_delta(delta + 1) + 1;    %record histogram
        end
        i = i + 1;
    end
end
%Normalize
p_delta = p_delta/(xc*yc);




end

