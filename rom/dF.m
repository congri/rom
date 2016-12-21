function [dF_dtheta_c, d2F_dtheta2] = dF(theta, theta_c, prior_type, prior_hyperparam, nTrain,...
    XNormSqMean, sumPhiTXmean, sumPhiSq, nCoarse)
%theta, sigma are the dependent variables, theta_c holds the current best estimate from EM
%Compute gradient and Hessian for posterior lower bound
%prior derivatives
if strcmp(prior_type, 'none')
    %no prior means derivatives are 0
    dprior_dthetac = 0;
    d2prior_d2thetac = 0;
else
[~, dprior_dthetac, d2prior_d2thetac] = log_prior_theta_c(theta, theta_c.theta, prior_type, prior_hyperparam);
end

%Compute sigma^-2 from dF/dSigma^-2 and plug it into the derivative dF/dtheta
sigma2 = (1/(nCoarse*nTrain))*(sum(XNormSqMean) - 2*theta'*sumPhiTXmean...
    + theta'*sumPhiSq*theta);
sigmaMinus2 = 1/sigma2;

%compute gradients of posterior lower bound
%dF/dsigma^-2 (prior independent of sigma)
% dL = (sigmaMinus2/nTrain)*(sumPhiTXmean - sumPhiSq*theta)
% dprior_dthetac

dF_dtheta_c = (sigmaMinus2/nTrain)*(sumPhiTXmean - sumPhiSq*theta) + dprior_dthetac;

%compute second derivatives
d2F_dtheta2 = -(sigmaMinus2/nTrain)*sumPhiSq + d2prior_d2thetac;

end

