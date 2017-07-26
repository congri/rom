function [k] = kernelFunction(precision, kernelOffset, type)
%Kernel functions to be used in p_c

if strcmp(type, 'squaredExponential')
    k = exp(-precision*kernelOffset^2);
else
    error('Unknown kernel functions')
end

end

