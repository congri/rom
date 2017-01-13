function [conductivity] = conductivityBackTransform(x, opts)
%Backtransformation from X to conductivity lambda

if strcmp(opts.transform, 'logit')
    %Logistic sigmoid transformation
    conductivity = (opts.upperCondLim - opts.lowerCondLim)./(1 + exp(-x)) + opts.lowerCondLim;
else
    error('unknown conductivity transformation')
end


end

