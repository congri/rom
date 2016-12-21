function [theta_c] = optTheta_c(theta_c, nTrain, nCoarse, XNormSqMean,...
    sumPhiTXmean, sumPhiSq, theta_prior_type, theta_prior_hyperparam,...
    sigma_prior_type, sigma_prior_hyperparam)
%% Find optimal theta_c and sigma

%levenberg-marquardt seems to be most stable
fsolve_options_theta = optimoptions('fsolve', 'SpecifyObjectiveGradient', true, 'Algorithm', 'levenberg-marquardt',...
    'Display', 'off');
fsolve_options_sigma = optimoptions('fsolve', 'SpecifyObjectiveGradient', true, 'Display', 'off');

%Solve self-consistently: compute optimal sigma2, then theta, then sigma2 again and so on
theta = theta_c.theta;
theta_old = theta;
sigma2 = theta_c.sigma^2;
logSigmaMinus2 = -2*log(theta_c.sigma);
iter = 0;
converged = false;
while(~converged)

    %Newton-Raphson maximization
    startValueTheta = theta;
    normGradientTol = 1e-5;
    provide_objective = false;
    debugNRmax = false;
    theta_old_old = theta;  %to check for iterative convergence
    ndiff = Inf;
    while(ndiff > 1e-20)
        gradHessTheta = @(theta) dF_dtheta(theta, sigma2, theta_old, theta_prior_type, theta_prior_hyperparam, nTrain,...
            sumPhiTXmean, sumPhiSq);
        theta_old = theta;
%         stepSizeTheta = .5;
%         theta = newtonRaphsonMaximization(gradHessTheta, startValueTheta,...
%             normGradientTol, provide_objective, stepSizeTheta, debugNRmax);
        diff = theta - theta_old;
        ndiff = norm(diff);
        theta = fsolve(gradHessTheta, startValueTheta, fsolve_options_theta);
    end
        
    %     theta = .5*theta + .5*theta_old;    %for stability
    
    gradHessLogSigmaMinus2 = @(lSigmaMinus2) dF_dlogSigmaMinus2(lSigmaMinus2, theta, nCoarse, nTrain, XNormSqMean,...
    sumPhiTXmean, sumPhiSq, sigma_prior_type, sigma_prior_hyperparam);
%     startValueLogSigmaMinus2 = logSigmaMinus2;
%     stepSizeSigma = .9; %the larger the faster, the smaller the more stable
%     logSigmaMinus2 = newtonRaphsonMaximization(gradHessLogSigmaMinus2, startValueLogSigmaMinus2,...
%         normGradientTol, provide_objective, stepSizeSigma, debugNRmax);
    logSigmaMinus2 = fsolve(gradHessLogSigmaMinus2, logSigmaMinus2, fsolve_options_sigma);
    
    
    sigmaMinus2 = exp(logSigmaMinus2);
    sigma2_old = sigma2;
    sigma2 = 1/sigmaMinus2;
    
    
    if sigma2 == 0
        warning('sigma2 == 0. Set it to small finite value')
        sigma2 = 1e-60;
    end
%     sigma2 = .5*sigma2 + .5*sigma2_old %for stability
    
    iter = iter + 1;
    if(iter > 100 || norm(theta_old_old - theta)/norm(theta) < 1e-8)
        converged = true;
    end
    
end
theta_c.theta = theta;
theta_c.sigma = sqrt(sigma2);

end

