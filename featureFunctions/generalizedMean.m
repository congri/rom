function [m, dm_dz] = generalizedMean(x, z)
%Computes the generalized mean of x to parameter z

if(size(x, 3) > 1)
    error('generalized mean only defined for vector/matrix input')
end

if z == 0
    if any(x == 0)
        m = 0;
    else
        m = exp(mean(mean(log(x))));
    end
else
    m = ((1/numel(x))*sum(sum(x.^z)))^(1/z);
end

if nargout > 1
    %Compute derivative w.r.t. parameter z
    if z == 0
        if any(x == 0)
            dm_dz = 0;
        else
            dm_dz = .5*m*(1/numel(x))*(sum(sum(log(x).^2)) - (1/numel(x))*(sum(sum(log(x))))^2);
        end
    else
        dm_dz = (m/z)*(sum(sum((x.^z).*log(x)))/sum(sum(x.^z))  - (1/z)*log(mean(mean(x.^z))));
    end
end

end

