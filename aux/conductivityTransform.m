function [x] = conductivityTransform(conductivity, opts)
%Backtransformation from X to conductivity lambda

if strcmp(opts.transform, 'logit')
    %Logistic sigmoid transformation
    offset = 1e-80; %for stability
    x = log(conductivity - opts.lowerCondLim + offset)...
        - log(opts.upperCondLim - conductivity + offset);
else
    error('unknown conductivity transformation')
end


end

