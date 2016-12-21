function [m] = generalizedMean(x, z)
%Computes the generalized mean of x to parameter z

if(size(x, 1) > 1 && size(x, 2) > 1)
    error('generalized mean only defined for vector input')
end

if z == 0
    if any(x) == 0
        m = 0;
    else
        m = exp(mean(log(x)));
    end
else
    m = ((1/length(x))*sum(x.^z))^(1/z);
end


end

