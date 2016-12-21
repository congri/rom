function [dF_dtheta_c, d2F_dtheta2] = dF_dtheta(theta, sigma2, theta_old, prior_type, prior_hyperparam, nTrain,...
    sumPhiTXmean, sumPhiSq)
%theta, sigma are the dependent variables, theta_c holds the current best estimate from EM
%Compute gradient and Hessian for posterior lower bound
%prior derivatives
if strcmp(prior_type, 'none')
    %no prior means derivatives are 0
    dprior_dthetac = 0;
    d2prior_d2thetac = 0;
else
[~, dprior_dthetac, d2prior_d2thetac] = log_prior_theta_c(theta, theta_old, prior_type, prior_hyperparam);
end

sigmaMinus2 = 1/sigma2;

%compute gradients of posterior lower bound
%dF/dsigma^-2 (prior independent of sigma)
% dL = (sigmaMinus2/nTrain)*(sumPhiTXmean - sumPhiSq*theta)
% dprior_dthetac
% pause

dF_dtheta_c = sigmaMinus2*(sumPhiTXmean - sumPhiSq*theta) + dprior_dthetac;

%compute second derivatives
d2F_dtheta2 = - sigmaMinus2*sumPhiSq + d2prior_d2thetac;


%THOSE ARE NOT THE TRUE DERIVATIVES!!! WE CANCEL THE FACTOR N SIGMA^2 IN HESS^-1 * GRAD IN 
%NEWTON RAPHSON FOR STABILITY. USE THE COMMENTED UPPER LINES FOR TRUE DERIVATIVES
% dF_dtheta_c = (sumPhiTXmean - sumPhiSq*theta) + nTrain*sigma2*dprior_dthetac;
% 
% %compute second derivatives
% d2F_dtheta2 = -sumPhiSq + nTrain*sigma2*d2prior_d2thetac;

end