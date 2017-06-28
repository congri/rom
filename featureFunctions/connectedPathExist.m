function [conPath] = connectedPathExist(lambdak, phase, phaseConductivity, dir, mode)
%Is there a connected path within phase phase from left to right (dir == x) or top to bottom (dir == y)?
%If yes, the output is 1 (or 1/min(dist), depending on mode), 0 if not

%Transform lambda to binary
lambdakBin = (lambdak == phaseConductivity(phase));

if(dir == 'x')
    %Check first if there are true elements on the relevant boundaries (left/right for x)
    left = lambdakBin(:, 1);
    right = lambdakBin(:, end);
    if(any(left) && any(right))
        %Loop through left boundary and use right boundary as mask for matlab bwgeodesic function
        leftIndex = find(left);
        conPath = 0;
        for i = leftIndex'
            geo = bwdistgeodesic(lambdakBin, 1, i);
            if(any(isfinite(geo(:, end))))
                if strcmp(mode, 'exist')
                    %Connected path found, exit loop
                    conPath = 1;
                    break;
                elseif strcmp(mode, 'invdist')
                    conPathTemp = 1/min(geo(:, end));
                    if(conPathTemp > conPath)
                        conPath = conPathTemp;
                        if(~isfinite(conPath))
                            i
                            dir
                            geo
                            error('Zero path length of connected path')
                        end
                    end
                end
            end
        end
    else
        conPath = 0;
    end
elseif(dir == 'y')
    %Check first if there are true elements on the relevant boundaries (top/bottom for y)
    top = lambdakBin(1, :);
    bottom = lambdakBin(end, :);
    if(any(top) && any(bottom))
        %Loop through upper boundary and use lower boundary as mask for matlab bwgeodesic function
        topIndex = find(top);
        conPath = 0;
        for i = topIndex
            geo = bwdistgeodesic(lambdakBin, i, 1);
            if(any(isfinite(geo(end, :))))
                if strcmp(mode, 'exist')
                    %Connected path found, exit loop
                    conPath = 1;
                    break;
                elseif strcmp(mode, 'invdist')
                    conPathTemp = 1/min(geo(end, :));
                    if(conPathTemp > conPath)
                        conPath = conPathTemp;
                        if(~isfinite(conPath))
                            i
                            dir
                            geo
                            error('Zero path length of connected path')
                        end
                    end
                end
            end
        end
    else
        conPath = 0;
    end
else
    error('which direction?')
end

end

