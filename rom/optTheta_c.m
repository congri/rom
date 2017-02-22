function [theta_c] = optTheta_c(theta_c, nTrain, nCoarse, XSqMean,...
    Phi, XMean, theta_prior_type, theta_prior_hyperparam,...
    sigma_prior_type, sigma_prior_hyperparam)
%Find optimal theta_c and sigma

%% set options for iterative methods
%levenberg-marquardt seems to be most stable
fsolve_options_theta = optimoptions('fsolve', 'SpecifyObjectiveGradient', true, 'Algorithm', 'trust-region-dogleg',...
    'Display', 'final-detailed', 'FunctionTolerance', 1e-50, 'StepTolerance', 1e-50, 'MaxIterations', 50);
fsolve_options_sigma = optimoptions('fsolve', 'SpecifyObjectiveGradient', true,...
    'Display', 'off', 'Algorithm', 'levenberg-marquardt');

dim_theta = numel(theta_c.theta);
XNormSqMean = sum(XSqMean);
sumXNormSqMean = sum(XNormSqMean);


%% Solve self-consistently: compute optimal sigma2, then theta, then sigma2 again and so on
theta = theta_c.theta;
I = speye(dim_theta);
% sigma2 = theta_c.sigma^2;
% sigma2 = 1e-8;  %start value
Sigma = 1e-6*speye(nCoarse);
% logSigmaMinus2 = -log(sigma2);

%sum_i Phi_i^T Sigma^-1 <X^i>_qi
sumPhiTSigmaInvXmean = 0;
SigmaInv = inv(Sigma);
SigmaInvXMean = Sigma\XMean;
sumPhiTSigmaInvPhi = 0;
PhiThetaMat = zeros(nCoarse, nTrain);

for i = 1:nTrain
    sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + Phi.designMatrices{i}'*SigmaInvXMean(:, i);
    sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + Phi.designMatrices{i}'*SigmaInv*Phi.designMatrices{i};
    PhiThetaMat(:, i) = Phi.designMatrices{i}*theta;
end

iter = 0;
converged = false;
while(~converged)

    theta_old_old = theta;  %to check for iterative convergence
    theta_old = theta;
    gradHessTheta = @(theta) dF_dtheta(theta, theta_old, theta_prior_type, theta_prior_hyperparam, nTrain,...
        sumPhiTSigmaInvXmean, sumPhiTSigmaInvPhi);
    
    if strcmp(theta_prior_type, 'hierarchical_laplace')
        %Matrix M is pos. def., invertible even if badly conditioned
%       warning('off', 'MATLAB:nearlySingularMatrix');
        offset = 1e-30;
        U = diag(sqrt((abs(theta_old) + offset)/theta_prior_hyperparam(1)));
    elseif strcmp(theta_prior_type, 'hierarchical_gamma')
        %Matrix M is pos. def., invertible even if badly conditioned
%       warning('off', 'MATLAB:nearlySingularMatrix');
        U = diag(sqrt((.5*abs(theta_old).^2 + theta_prior_hyperparam(2))./(theta_prior_hyperparam(1) + .5)));
    elseif strcmp(theta_prior_type, 'none')

    else
        error('Unknown prior on theta_c')
    end
    
    if strcmp(theta_prior_type, 'none')
        theta_temp = sumPhiTSigmaInvPhi\sumPhiTSigmaInvXmean;
    else
%         theta_temp = U*((sigma2*I + U*Phi.sumPhiTPhi*U)\U)*sumPhiTSigmaInvXmean;
        theta_temp = U*((U*sumPhiTSigmaInvPhi*U + I)\U)*sumPhiTSigmaInvXmean;
    end
    
    %Catch instabilities
    if(norm(theta_temp)/length(theta_temp) > 5e1 || any(~isfinite(theta_temp)))
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
    
    %     theta = .5*theta + .5*theta_old;    %for stability
    
    if strcmp(sigma_prior_type, 'none')
        %         sigma2 = (1/(nTrain*nCoarse))*(sumXNormSqMean - 2*theta'*sumPhiTSigmaInvXmean + theta'*Phi.sumPhiTPhi*theta);
        Sigma = sparse(1:nCoarse, 1:nCoarse, mean(XSqMean - 2*(PhiThetaMat.*XMean) + PhiThetaMat.^2, 2));
        
        sigma2CutoffHi = 4;
        sigma2CutoffLo = eps;
        if any(diag(Sigma) < sigma2CutoffLo)
            warning('sigma2 < cutoff. Set it to small cutoff value')
            %         sigma2 = sigma2CutoffLo;
            s = diag(Sigma);
            s(s < sigma2CutoffLo) = sigma2CutoffLo;
            index = 1:nCoarse;
            Sigma = sparse(index, index, s);
        elseif any(any(Sigma > sigma2CutoffHi))
            warning('sigma2 > cutoff, set it to cutoff')
            Sigma(Sigma > sigma2CutoffHi) = sigma2CutoffHi;
        end
        
        %sum_i Phi_i^T Sigma^-1 <X^i>_qi
        sumPhiTSigmaInvXmean = 0;
        %Only valid for diagonal Sigma
        s = diag(Sigma);
        SigmaInv = sparse(diag(1./s));
        SigmaInvXMean = Sigma\XMean;
        sumPhiTSigmaInvPhi = 0;
        PhiThetaMat = zeros(nCoarse, nTrain);
        
        for i = 1:nTrain
            sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + Phi.designMatrices{i}'*SigmaInvXMean(:, i);
            sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + Phi.designMatrices{i}'*SigmaInv*Phi.designMatrices{i};
            PhiThetaMat(:, i) = Phi.designMatrices{i}*theta;
        end
    else
        error('Prior on diagonal Sigma not yet implemented')
        gradHessLogSigmaMinus2 = @(lSigmaMinus2) dF_dlogSigmaMinus2(lSigmaMinus2, theta, nCoarse, nTrain, XNormSqMean,...
            sumPhiTSigmaInvXmean, Phi.sumPhiTPhi, sigma_prior_type, sigma_prior_hyperparam);
        %     startValueLogSigmaMinus2 = logSigmaMinus2;
        %     stepSizeSigma = .9; %the larger the faster, the smaller the more stable
        %     logSigmaMinus2 = newtonRaphsonMaximization(gradHessLogSigmaMinus2, startValueLogSigmaMinus2,...
        %         normGradientTol, provide_objective, stepSizeSigma, debugNRmax);
        logSigmaMinus2 = fsolve(gradHessLogSigmaMinus2, logSigmaMinus2, fsolve_options_sigma);
        sigmaMinus2 = exp(logSigmaMinus2);
        sigma2_old = sigma2;
        sigma2 = 1/sigmaMinus2;
    end
    
    iter = iter + 1;
    thetaDiffRel = norm(theta_old_old - theta)/(norm(theta)*numel(theta));
    if((iter > 20 && thetaDiffRel < 1e-8) || iter > 200)
        converged = true;
    end
    
end
theta_c.theta = theta;
% theta_c.sigma = sqrt(sigma2);
theta_c.Sigma = Sigma;

end

