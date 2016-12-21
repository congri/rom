function [log_p, d_log_p, d2_log_p] = log_prior_theta_c(theta_c, theta_c_old, prior_type, prior_hyperparam)
%Gives the prior log probability and derivative of parameter theta_c
% theta_c is a column vector
% mode gives the functional form of the prior

dim = size(theta_c, 1);
offset = 1e-10;
if strcmp(prior_type, 'gaussian')
    
    %Gaussian prior
    %hyperparameters
    mu = 0*theta_c;
    Sigma = prior_hyperparam*eye(length(theta_c));
    
%     log_p = -.5*dim*log(2*pi) - .5*logdet(Sigma) - .5*(theta_c - mu)'*(Sigma\(theta_c - mu));
    log_p = NaN;    %We do not compute this as it is not needed to maximize posterior
    d_log_p = - Sigma\(theta_c - mu);
    if nargout > 2
       d2_log_p = - inv(Sigma); 
    end
    
elseif strcmp(prior_type, 'spikeAndSlab')
    
    %Sum of two zero mean Gaussians, one with epsilon variance
    %MIGHT BE NUMERICALLY UNSTABLE DUE TO LOG!!!
    %notation
    spikeWeight = prior_hyperparam(1);
    spikeVar = prior_hyperparam(2); %typically very small for sparsity
    slabVar = prior_hyperparam(3);  %slabVar >> spikeVar
    
    q1 = spikeWeight*normpdf(theta_c_old, 0, sqrt(spikeVar));
    q2 = (1 - spikeWeight)*normpdf(theta_c_old, 0, sqrt(slabVar));
    Zq = q1 + q2;
    q1 = q1./Zq;
    q2 = q2./Zq;
    log_p = NaN;    %We do not implement this as it is not needed for max posterior
    d_log_p = -(q1/spikeVar + q2/slabVar).*theta_c;
    if nargout > 2
        d2_log_p = - diag(q1/spikeVar + q2/slabVar);
    end
    
elseif strcmp(prior_type, 'laplace')
    
    %Laplacian prior
    log_p = dim*log(prior_hyperparam) - dim*log(2) - prior_hyperparam*sum(abs(theta_c));
    d_log_p = - prior_hyperparam*sign(theta_c);  
    if nargout > 2
       d2_log_p = zeros(dim);
    end
    
elseif strcmp(prior_type, 'hierarchical_laplace')
    
    log_p = - prior_hyperparam(1)*sum((theta_c.^2)./(abs(theta_c_old) + offset));
    d_log_p = - 2*prior_hyperparam(1)*theta_c./(abs(theta_c_old) + offset);
    if nargout > 2
       d2_log_p = - 2*prior_hyperparam(1)*diag(1./(abs(theta_c_old) + offset));
    end
    
elseif strcmp(prior_type, 'hierarchical_gamma')
    
    %Hierarchical Bayesian model with gamma hyperprior on precision
    log_p = -.5*sum((theta_c.^2).*((prior_hyperparam(1) + .5)./(.5*theta_c_old.^2 + prior_hyperparam(2))));
    d_log_p = - theta_c.*((prior_hyperparam(1) + .5)./(.5*theta_c_old.^2 + prior_hyperparam(2)));
    if nargout > 2
       d2_log_p = - diag(((prior_hyperparam(1) + .5)./(.5*theta_c_old.^2 + prior_hyperparam(2))));
    end
    
else
    error('Unknown prior for theta_c')
end

end

