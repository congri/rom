function [lg_prior_sigma, d_lg_prior_sigma, d2_lg_prior_sigma] = log_prior_sigma(sigma2, prior_type, prior_param)
%Derivative of prior on sigma w.r.t. sigma^-2

if strcmp(prior_type, 'exponential_sigma2')
    lg_prior_sigma = log(prior_param) - prior_param*sigma2;
    if nargout > 1
        d_lg_prior_sigma = prior_param*sigma2^2;
    end
    if nargout > 2
        d2_lg_prior_sigma = -2*prior_param*sigma2^3;
    end
elseif strcmp(prior_type, 'exponential_sigma')
    sigma = sqrt(sigma2);
    lg_prior_sigma = log(prior_param) - prior_param*sigma;
    if nargout > 1
        d_lg_prior_sigma = .5*prior_param*sigma^3;
    end
    if nargout > 2
        d2_lg_prior_sigma = -.75*prior_param*sigma^5;
    end
else
    error('Unknown prior on sigma')
end

end

