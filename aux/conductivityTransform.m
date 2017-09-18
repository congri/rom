function [x] = conductivityTransform(conductivity, opts)
%Transformation from conductivity lambda to x

if strcmp(opts.type, 'logit')
    %Logistic sigmoid transformation
    offset = 1e-80; %for stability
    %     x = log(conductivity - opts.lowerCondLim + offset)...
    %         - log(opts.upperCondLim - conductivity + offset) + opts.shift;
    x = - log((opts.limits(2) - opts.limits(1))./(conductivity - opts.limits(1)) - 1 + offset);
elseif strcmp(opts.type, 'log')
    offset = realmin;
    x = log(conductivity + offset);
elseif strcmp(opts.type, 'log_lower_bound')
    offset = 1e-80;
    x = log(conductivity - opts.lowerCondLim + offset);
elseif strcmp(opts.type, 'log_cholesky')
    %log Cholesky decomposition, enables anisotropy in coarse conductivity elements
    error('log Cholesky transform not implemented')
elseif strcmp(opts.type, 'square')
    x = sqrt(conductivity);
else
    error('unknown conductivity transformation')
end


end

