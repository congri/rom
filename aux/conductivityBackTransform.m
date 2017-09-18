function [conductivity] = conductivityBackTransform(x, opts)
%Backtransformation from X to conductivity lambda

%For log_cholesky: x must be 3 x N, where the rows give log(l1), l2 and log(l3), the params of the
%cholesky matrix L = [l1, l2; 0, l3]
%The output is then a 3-dimensional array, holding the 2x2 conductivity tensors in the first 2
%indices and the 3rd index runs from 1 to N, i.e. input size of x

if strcmp(opts.type, 'logit')
    %Logistic sigmoid transformation
    conductivity = (opts.limits(2) - opts.limits(1))./(1 + exp(-x)) + opts.limits(1);
elseif strcmp(opts.type, 'log_cholesky')
    %conductivity is a 2x2 tensor now, store in 3-dim array
    N = size(x, 2);
    if(~(size(x, 1) == 3))
        error('X must contain 3 parameters for each conductivity tensor')
    end
    conductivity = zeros(2, 2, N);
    for i = 1:N
        %cholesky matrix
        L = [exp(x(1, i)), x(2, i); 0, exp(x(3, i))];
        conductivity(:, :, i) = L'*L;
    end
elseif strcmp(opts.type, 'log')
    conductivity = exp(x);
    if any(any(conductivity < opts.limits(1)))
        %lower bound on conductivity for stability
        conductivity(conductivity < opts.limits(1)) = opts.limits(1);
%         warning('Effective conductivity is below lower bound')
    end
    if any(any(conductivity > opts.limits(2)))
        %upper bound on conductivity for stability
        conductivity(conductivity > opts.limits(2)) = opts.limits(2);
%         warning('Effective conductivity is above upper bound')
    end
elseif strcmp(opts.type, 'log_lower_bound')
    %log transformation, where the eff. cond. lower bound is non-zero 
    conductivity = exp(x) + opts.limits(1);
    if any(any(conductivity > opts.limits(2)))
        %upper bound on conductivity for stability
        conductivity(conductivity > opts.limits(2)) = opts.limits(2);
        warning('Effective conductivity is above upper bound')
    end
elseif strcmp(opts.type, 'square')
    conductivity = x.^2;
    if any(any(conductivity < opts.limits(1)))
        %lower bound on conductivity for stability
        conductivity(conductivity < opts.limits(1)) = opts.limits(1);
%         warning('Effective conductivity is below lower bound')
    end
    if any(any(conductivity > opts.limits(2)))
        %upper bound on conductivity for stability
        conductivity(conductivity > opts.limits(2)) = opts.limits(2);
%         warning('Effective conductivity is above upper bound')
    end
else
    error('unknown conductivity transformation')
end

if(any(any(~isfinite(conductivity))))
    warning('Non-finite conductivity')
    conductivity
    x
    conductivity(~isfinite(conductivity)) = .5*(opts.limits(1) + opts.limits(2));
end
end

