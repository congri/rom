function [colnorm] = columnnorm(array)
%Computes norm of columns of array


sizeOfArray = size(array);

if numel(sizeOfArray) == 2
    colnorm = zeros(1, sizeOfArray(2));
    for i = 1:sizeOfArray(2)
        colnorm(i) = norm(array(:, i));
    end
elseif numel(sizeOfArray) == 3
    colnorm = zeros(1, sizeOfArray(2), sizeOfArray(3));
    for i = 1:sizeOfArray(2)
        for j = 1:sizeOfArray(3)
            colnorm(1, i, j) = norm(array(:, i, j));
        end
    end
else
    error('columnnorm not implemented for more than 3D arrays')
end

end