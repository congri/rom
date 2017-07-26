function [theta_c] = optTheta_cKernels(theta_c, nTrain, nCoarse, XSqMean,...
    Phi, XMean, sigma_prior_type, sigma_prior_hyperparam)
%Find optimal theta_c and sigma

dim_theta = numel(theta_c.theta);

%% Solve self-consistently: compute optimal sigma2, then theta, then sigma2 again and so on
theta = theta_c.theta;  %kernel coefficients
nu = theta_c.nu;        %kernel locations
tau = theta_c.tau;      %kernel bandwidths
Sigma = theta_c.Sigma;
I = speye(dim_theta);

%sum_i Phi_i^T Sigma^-1 <X^i>_qi
sumKTSigmaInvXmean = 0;
SigmaInv = theta_c.SigmaInv;
SigmaInvXMean = SigmaInv*XMean;
sumKTSigmaInvK = 0;
KThetaMat = zeros(nCoarse, nTrain);

kernelMatrix{1} = zeros(nCoarse, theta_c.nKernels);  %prealloc
kernelMatrix = repmat(kernelMatrix, nTrain, 1);
kernelOffset = kernelMatrix;
for n = 1:nTrain
    for m = 1:theta_c.nKernels
        for k = 1:nCoarse
            kernelOffset{n}(k, m) = norm(nu(m, :) - Phi.designMatrices{n}(k, :))^2;
            kernelMatrix{n}(k, m) = kernelFunction(tau(m), kernelOffset{n}(k, m), 'squaredExponential');
        end
    end
end



for n = 1:nTrain
    sumKTSigmaInvXmean = sumKTSigmaInvXmean + kernelMatrix{n}'*SigmaInvXMean(:, n);
    sumKTSigmaInvK = sumKTSigmaInvK + kernelMatrix{n}'*SigmaInv*kernelMatrix{n};
    KThetaMat(:, n) = kernelMatrix{n}*theta;
end

if(strcmp(theta_c.thetaPriorType, 'gaussian') || strcmp(theta_c.thetaPriorType, 'RVM'))
    %Find prior hyperparameter
    converged = false;
    iter = 0;
    while(~converged)
        stabilityParam = 1e-12;    %for stability in matrix inversion
        if strcmp(theta_c.thetaPriorType, 'gaussian')
            SigmaTilde = inv(sumKTSigmaInvK + (theta_c.priorHyperparam + stabilityParam)*I);
            muTilde = SigmaTilde*sumKTSigmaInvXmean;
            theta_prior_hyperparam_old = theta_c.priorHyperparam;
            theta_c.priorHyperparam = dim_theta/(muTilde'*muTilde + trace(SigmaTilde));
            if(abs(theta_c.priorHyperparam - theta_prior_hyperparam_old)/abs(theta_c.priorHyperparam) < 1e-5...
                    || iter > 100)
                converged = true;
            elseif(~isfinite(theta_c.priorHyperparam) || theta_c.priorHyperparam <= 0)
                converged = true;
                theta_c.priorHyperparam = 1;
                warning('Gaussian hyperparameter precision is negative or not a number. Setting it to 1.')
            end
        elseif strcmp(theta_c.thetaPriorType, 'RVM')
            SigmaTilde = inv(sumKTSigmaInvK + diag(theta_c.priorHyperparam + stabilityParam));
            muTilde = SigmaTilde*sumKTSigmaInvXmean;
            theta_prior_hyperparam_old = theta_c.priorHyperparam;
            theta_c.priorHyperparam = 1./(muTilde.^2 + diag(SigmaTilde));
            if(norm(theta_c.priorHyperparam - theta_prior_hyperparam_old)/norm(theta_c.priorHyperparam) < 1e-5...
                    || iter > 100)
                converged = true;
            elseif(any(~isfinite(theta_c.priorHyperparam)) || any(theta_c.priorHyperparam <= 0))
                converged = true;
                theta_c.priorHyperparam = ones(dim_theta, 1);
                warning('Gaussian hyperparameter precision is negative or not a number. Setting it to 1.')
            end
        end
        iter = iter + 1;
    end
end

iter = 0;
converged = false;
while(~converged)
    theta_old_old = theta;  %to check for iterative convergence
    theta_old = theta;
    
    if strcmp(theta_c.thetaPriorType, 'hierarchical_laplace')
        %Matrix M is pos. def., invertible even if badly conditioned
        %       warning('off', 'MATLAB:nearlySingularMatrix');
        offset = 1e-30;
        U = diag(sqrt((abs(theta_old) + offset)/theta_c.priorHyperparam(1)));
    elseif strcmp(theta_c.thetaPriorType, 'hierarchical_gamma')
        %Matrix M is pos. def., invertible even if badly conditioned
        %       warning('off', 'MATLAB:nearlySingularMatrix');
        U = diag(sqrt((.5*abs(theta_old).^2 + theta_c.priorHyperparam(2))./(theta_c.priorHyperparam(1) + .5)));
    elseif strcmp(theta_c.thetaPriorType, 'gaussian')
        sumKTSigmaInvK = sumKTSigmaInvK + (theta_c.priorHyperparam(1) + stabilityParam)*I;
    elseif strcmp(theta_c.thetaPriorType, 'RVM')
        sumKTSigmaInvK = sumKTSigmaInvK + diag(theta_c.priorHyperparam + stabilityParam);
    elseif strcmp(theta_c.thetaPriorType, 'none')
        
    else
        error('Unknown prior on theta_c')
    end
    
    if (strcmp(theta_c.thetaPriorType, 'gaussian') || strcmp(theta_c.thetaPriorType, 'RVM') ||...
            strcmp(theta_c.thetaPriorType, 'none'))
        theta_temp = sumKTSigmaInvK\sumKTSigmaInvXmean;
    else
        %         theta_temp = U*((sigma2*I + U*Phi.sumPhiTPhi*U)\U)*sumPhiTSigmaInvXmean;
        theta_temp = U*((U*sumKTSigmaInvK*U + I)\U)*sumKTSigmaInvXmean;
    end
    [~, msgid] = lastwarn;     %to catch nearly singular matrix

    if(strcmp(msgid, 'MATLAB:singularMatrix') || strcmp(msgid, 'MATLAB:nearlySingularMatrix')...
            || strcmp(msgid, 'MATLAB:illConditionedMatrix') || norm(theta_temp)/length(theta_temp) > 10)
        warning('theta_c is assuming unusually large values. Only go small step.')
        %         theta = fsolve(gradHessTheta, theta, fsolve_options_theta);
%         theta = .95*theta_old + .05*norm(theta_old)*theta_temp/norm(theta_temp)
        theta = normrnd(theta_old, 1);
        if any(~isfinite(theta))
            %restart from 0
            warning('Some components of theta are not finite. Restarting from random theta...')
            theta = normrnd(theta_old, 1);
        end
    else
        theta = theta_temp;
    end
    
    if(norm(theta) > 100)
        pause
    end
    iter
    theta
    theta_temp
    gradfun = @(NU) dF_dnu(NU, SigmaInv, XMean, theta, tau, Phi.designMatrices, theta_c.nKernels);
    
    nuOptions = optimoptions(@fminunc, 'Display', 'iter', 'SpecifyObjectiveGradient', true);
    [nu, ~, exitflag] = fminunc(gradfun, nu, nuOptions)
    if(exitflag == 5)
        nuOptions = optimset('Display', 'iter');
        [nu, ~, exitflag2] = fminsearch(gradfun, nu, nuOptions)
    end

%     options.Xtol = 1e-6;
%     options.gradTol = 1e-6;
%     options.initialStepSize = .01;
%     options.debug = false;
%     options.provide_objective = false;
%     [nu, grad, nIter] = BFGSMaximization(gradfun, nu, options)
%     pause
    
    for n = 1:nTrain
        for m = 1:theta_c.nKernels
            for k = 1:nCoarse
                kernelOffset{n}(k, m) = norm(nu(m, :) - Phi.designMatrices{n}(k, :))^2;
                kernelMatrix{n}(k, m) = kernelFunction(tau(m), kernelOffset{n}(k, m), 'squaredExponential');
            end
        end
    end
        
    for n = 1:nTrain
        sumKTSigmaInvXmean = sumKTSigmaInvXmean + kernelMatrix{n}'*SigmaInvXMean(:, n);
        sumKTSigmaInvK = sumKTSigmaInvK + kernelMatrix{n}'*SigmaInv*kernelMatrix{n};
        KThetaMat(:, n) = kernelMatrix{n}*theta;
    end
    
    KThetaMat = zeros(nCoarse, nTrain);
    for n = 1:nTrain
        KThetaMat(:, n) = kernelMatrix{n}*theta;
    end
        
    if strcmp(sigma_prior_type, 'none')
        Sigma = sparse(1:nCoarse, 1:nCoarse, mean(XSqMean - 2*(KThetaMat.*XMean) + KThetaMat.^2, 2));
        Sigma(Sigma < 0) = eps; %for numerical stability
        
        %sum_i Phi_i^T Sigma^-1 <X^i>_qi
        sumKTSigmaInvXmean = 0;
        %Only valid for diagonal Sigma
        s = diag(Sigma);
        SigmaInv = sparse(diag(1./s));
        SigmaInvXMean = SigmaInv*XMean;
        sumKTSigmaInvK = 0;
        
        for n = 1:nTrain
            sumKTSigmaInvXmean = sumKTSigmaInvXmean + kernelMatrix{n}'*SigmaInvXMean(:, n);
            sumKTSigmaInvK = sumKTSigmaInvK + kernelMatrix{n}'*SigmaInv*kernelMatrix{n};
        end
    elseif strcmp(sigma_prior_type, 'expSigSq')
        %Vector of variances
        sigmaSqVec = .5*(1./sigma_prior_hyperparam).*...
            (-.5*nTrain + sqrt(2*nTrain*sigma_prior_hyperparam.*...
            mean(XSqMean - 2*(KThetaMat.*XMean) + KThetaMat.^2, 2) + .25*nTrain^2));
        
        Sigma = sparse(1:nCoarse, 1:nCoarse, sigmaSqVec);
        Sigma(Sigma < 0) = eps; %for numerical stability
        sumKTSigmaInvXmean = 0;
        %Only valid for diagonal Sigma
        s = diag(Sigma);
        SigmaInv = sparse(diag(1./s));
        SigmaInvXMean = SigmaInv*XMean;
        sumKTSigmaInvK = 0;
        
        for n = 1:nTrain
            sumKTSigmaInvXmean = sumKTSigmaInvXmean + kernelMatrix{n}'*SigmaInvXMean(:, n);
            sumKTSigmaInvK = sumKTSigmaInvK + kernelMatrix{n}'*SigmaInv*kernelMatrix{n};
        end        
    elseif strcmp(sigma_prior_type, 'delta')
        %Don't change sigma
        %sum_i Phi_i^T Sigma^-1 <X^i>_qi
        sumKTSigmaInvXmean = 0;
        %Only valid for diagonal Sigma
        s = diag(Sigma);
        SigmaInv = sparse(diag(1./s));
        SigmaInvXMean = SigmaInv*XMean;
        sumKTSigmaInvK = 0;
        
        for n = 1:nTrain
            sumKTSigmaInvXmean = sumKTSigmaInvXmean + kernelMatrix{n}'*SigmaInvXMean(:, n);
            sumKTSigmaInvK = sumKTSigmaInvK + kernelMatrix{n}'*SigmaInv*kernelMatrix{n};
        end
    end
    iter = iter + 1;
    thetaDiffRel = norm(theta_old_old - theta)/(norm(theta)*numel(theta));
    if((iter > 5 && thetaDiffRel < 1e-8) || iter > 20)
        converged = true;
    end
end

theta_c.theta = theta;
theta_c.nu = nu;
theta_c.tau = tau;
% theta_c.sigma = sqrt(sigma2);
theta_c.Sigma = Sigma;
theta_c.SigmaInv = SigmaInv;
theta_c.priorHyperparam = theta_c.priorHyperparam;
%Value which could be used to refine mesh
noPriorSigma = mean(XSqMean - 2*(KThetaMat.*XMean) + KThetaMat.^2, 2);
save('./data/noPriorSigma.mat', 'noPriorSigma');

end




