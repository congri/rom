function [centers] = elementCenters(gridVectorX, gridVectorY)
%Computes centers of coarse elements
%gridVectorX, gridVectorY give element widths in x- and y-direction


cumsumX = cumsum(gridVectorX);
cumsumY = cumsum(gridVectorY);

centersX = cumsumX - gridVectorX/2;
centersY = cumsumY - gridVectorY/2;

centers = zeros(2, length(centersX)*length(centersY));

k = 1;
for i = 1:length(gridVectorX)
    for j = 1:length(gridVectorY)
        centers(:, k) = [centersX(i); centersY(j)];
        k = k + 1;
    end
end

end

