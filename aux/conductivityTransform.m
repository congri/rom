function [x] = conductivityTransform(conductivity, opts)
%Transformation from conductivity lambda to x

if strcmp(opts.transform, 'logit')
    %Logistic sigmoid transformation
    offset = 1e-80; %for stability
    %     x = log(conductivity - opts.lowerCondLim + offset)...
    %         - log(opts.upperCondLim - conductivity + offset) + opts.shift;
    x = - log((opts.upperCondLim - opts.lowerCondLim)./(conductivity - opts.lowerCondLim) - 1 + offset);
elseif strcmp(opts.transform, 'log')
    offset = realmin;
    x = log(conductivity + offset);
elseif strcmp(opts.transform, 'log_lower_bound')
    offset = 1e-80;
    x = log(conductivity - opts.lowerCondLim + offset);
elseif strcmp(opts.transform, 'log_cholesky')
    %log Cholesky decomposition, enables anisotropy in coarse conductivity elements
    error('log Cholesky transform not implemented')
else
    error('unknown conductivity transformation')
end


end

