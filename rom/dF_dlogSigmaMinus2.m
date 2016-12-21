function [dF_dlogSigmaMinus2, d2F_dlogSigmaMinus22] = dF_dlogSigmaMinus2(logSigmaMinus2, theta, nCoarse, nTrain, XNormSqMean,...
    sumPhiTXmean, sumPhiSq, prior_type, prior_param)
%Gradient and Hessian of EM lower bound w.r.t. log sigma^-2

sigmaMinus2 = exp(logSigmaMinus2);
if isinf(sigmaMinus2)
    warning('sigma^-2 is infinite, set it to 1e30')
    stabex = 30;
    sigmaMinus2 = 10^stabex;
    sigmaMinus4 = 10^(2*stabex);
    sigma2 = 10^(-stabex);
    sigma4 = 10^(-2*stabex);
else
    sigma2 = 1/sigmaMinus2;
    sigma4 = sigma2^2;
    sigmaMinus4 = sigmaMinus2^2;
end



if strcmp(prior_type, 'none')
    dlogprior_dsigmaMinus2 = 0;
    d2logprior_dsigmaMinus22 = 0;
else
    [~, dlogprior_dsigmaMinus2, d2logprior_dsigmaMinus22] = log_prior_sigma(sigma2, prior_type, prior_param);
end

dF_dsigmaMinus2 = .5*nCoarse*nTrain*sigma2 - .5*(sum(XNormSqMean) - 2*theta'*sumPhiTXmean...
        + theta'*sumPhiSq*theta) + dlogprior_dsigmaMinus2;
d2F_dsigmaMinus22 = -.5*nCoarse*nTrain*sigma4 + d2logprior_dsigmaMinus22;

%USE THESE ONES IF YOU WANT TRUE GRAD W.R.T. LOG SIGMA^-2!!!
dF_dlogSigmaMinus2 = sigmaMinus2*dF_dsigmaMinus2;
d2F_dlogSigmaMinus22 = dF_dlogSigmaMinus2 + (sigmaMinus4)*d2F_dsigmaMinus22;

%We use these derivatives for Newton-Raphson, where we need only the ratio
%dF_dlogSigmaMinus2/d2F_dlogSigmaMinus22. Hence we can cancel common factors
% dF_dlogSigmaMinus2 = dF_dsigmaMinus2;
% d2F_dlogSigmaMinus22 = dF_dsigmaMinus2 -.5*nCoarse*sigma2 + sigmaMinus2*d2logprior_dsigmaMinus22;

end

