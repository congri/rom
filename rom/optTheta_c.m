function [theta_c] = optTheta_c(theta_c, nTrain, nCoarse, XNormSqMean,...
    sumPhiTXmean, sumPhiSq, theta_prior_type, theta_prior_hyperparam,...
    sigma_prior_type, sigma_prior_hyperparam)
%% Find optimal theta_c and sigma

%levenberg-marquardt seems to be most stable
fsolve_options_theta = optimoptions('fsolve', 'SpecifyObjectiveGradient', true, 'Algorithm', 'trust-region-dogleg',...
    'Display', 'final-detailed', 'FunctionTolerance', 1e-50, 'StepTolerance', 1e-50, 'MaxIterations', 50);
fsolve_options_sigma = optimoptions('fsolve', 'SpecifyObjectiveGradient', true,...
    'Display', 'off', 'Algorithm', 'levenberg-marquardt');

%Solve self-consistently: compute optimal sigma2, then theta, then sigma2 again and so on
theta = theta_c.theta;
I = eye(length(theta));
sigma2 = theta_c.sigma^2;
logSigmaMinus2 = -2*log(theta_c.sigma);
iter = 0;
converged = false;
while(~converged)

    theta_old_old = theta;  %to check for iterative convergence
    ndiff = Inf;
    i = 1;
    %Inner EM-loop - obsolete?
    while(ndiff > 1e-20 && i < 2)
        theta_old = theta;
        gradHessTheta = @(theta) dF_dtheta(theta, sigma2, theta_old, theta_prior_type, theta_prior_hyperparam, nTrain,...
            sumPhiTXmean, sumPhiSq);
        
        if strcmp(theta_prior_type, 'hierarchical_laplace')
            %Matrix M is pos. def., invertible even if badly conditioned
%             warning('off', 'MATLAB:nearlySingularMatrix');
%             M = sumPhiSq + theta_prior_hyperparam(1)*diag((2*sigma2)./(abs(theta_old) + offset));
            offset = 1e-30;
%             V = theta_prior_hyperparam(1)*diag(1./(abs(theta_old) + offset));
            U = diag(sqrt((abs(theta_old) + offset)/theta_prior_hyperparam(1)));
        elseif strcmp(theta_prior_type, 'hierarchical_gamma')
            %Matrix M is pos. def., invertible even if badly conditioned
%             warning('off', 'MATLAB:nearlySingularMatrix');
%             M = sumPhiSq + sigma2*diag((theta_prior_hyperparam(1) + .5)./(.5*abs(theta_old).^2 + theta_prior_hyperparam(2)));
%             V = diag((theta_prior_hyperparam(1) + .5)./(.5*abs(theta_old).^2 + theta_prior_hyperparam(2)));
            U = diag(sqrt((.5*abs(theta_old).^2 + theta_prior_hyperparam(2))./(theta_prior_hyperparam(1) + .5)));
        elseif strcmp(theta_prior_type, 'none')
            M = sumPhiSq;
        else
            error('Unknown prior on theta_c')
        end

        if strcmp(theta_prior_type, 'none')
            theta_temp = sumPhiSq\sumPhiTXmean;
        else
%             theta_temp = M\sumPhiTXmean;
%             theta_temp = (sigma2*V + sumPhiSq)\sumPhiTXmean;
            theta_temp = U*((sigma2*I + U*sumPhiSq*U)\U)*sumPhiTXmean;
        end
        
        if(norm(theta_temp)/length(theta_temp) > 5e1)
            warning('theta_c is assuming unusually large values. Using Newton-Raphson instead of mldivide.')
            %theta = fsolve(gradHessTheta, theta, fsolve_options_theta);
            
            %Newton-Raphson maximization
            startValueTheta = theta;
            normGradientTol = eps;
            provide_objective = false;
            debugNRmax = false;
            RMMode = false;
            stepSizeTheta = .6;
            theta = newtonRaphsonMaximization(gradHessTheta, startValueTheta,...
                normGradientTol, provide_objective, stepSizeTheta, RMMode, debugNRmax);
        else
            theta = theta_temp;
        end
        diff = theta - theta_old;
        ndiff = norm(diff);
        i = i + 1;
    end
        
    %     theta = .5*theta + .5*theta_old;    %for stability
    
    if strcmp(sigma_prior_type, 'none')
        sigma2 = (1/(nTrain*nCoarse))*(sum(XNormSqMean) - 2*theta'*sumPhiTXmean + theta'*sumPhiSq*theta);
    else
        gradHessLogSigmaMinus2 = @(lSigmaMinus2) dF_dlogSigmaMinus2(lSigmaMinus2, theta, nCoarse, nTrain, XNormSqMean,...
            sumPhiTXmean, sumPhiSq, sigma_prior_type, sigma_prior_hyperparam);
        %     startValueLogSigmaMinus2 = logSigmaMinus2;
        %     stepSizeSigma = .9; %the larger the faster, the smaller the more stable
        %     logSigmaMinus2 = newtonRaphsonMaximization(gradHessLogSigmaMinus2, startValueLogSigmaMinus2,...
        %         normGradientTol, provide_objective, stepSizeSigma, debugNRmax);
        logSigmaMinus2 = fsolve(gradHessLogSigmaMinus2, logSigmaMinus2, fsolve_options_sigma);
        sigmaMinus2 = exp(logSigmaMinus2);
        sigma2_old = sigma2;
        sigma2 = 1/sigmaMinus2
    end
    
    
    sigma2CutoffHi = 1e4;
    sigma2CutoffLo = 1e-12;
    if sigma2 < sigma2CutoffLo
        warning('sigma2 < cutoff. Set it to small cutoff value')
        sigma2 = sigma2CutoffLo;
    elseif sigma2 > sigma2CutoffHi
        warning('sigma2 > cutoff, set it to cutoff')
        sigma2 = sigma2CutoffHi;
    end
    
    iter = iter + 1;
    thetaDiffRel = norm(theta_old_old - theta)/(norm(theta)*numel(theta));
    if((iter > 20 && thetaDiffRel < 1e-8) || iter > 200)
        converged = true;
    end
    
end
theta_c.theta = theta;
theta_c.sigma = sqrt(sigma2);

end

