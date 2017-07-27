function [theta_c] = optTheta_c(theta_c, nTrain, nCoarse, XSqMean,...
    designMatrix, XMean, sigma_prior_type, sigma_prior_hyperparam)
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
% theta = .1*ones(size(theta_c.theta));
% theta = 2*rand(size(theta_c.theta)) - 1;
theta = theta_c.theta;
I = speye(dim_theta);
% sigma2 = theta_c.sigma^2;
% sigma2 = 1e-8;  %start value
% Sigma = 1e-6*speye(nCoarse);
Sigma = theta_c.Sigma;
% logSigmaMinus2 = -log(sigma2);

%sum_i Phi_i^T Sigma^-1 <X^i>_qi
sumPhiTSigmaInvXmean = 0;
% SigmaInv = inv(Sigma);
% SigmaInvXMean = Sigma\XMean;
SigmaInv = theta_c.SigmaInv;
SigmaInvXMean = SigmaInv*XMean;
sumPhiTSigmaInvPhi = 0;
PhiThetaMat = zeros(nCoarse, nTrain);

for i = 1:nTrain
    sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + designMatrix{i}'*SigmaInvXMean(:, i);
    sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + designMatrix{i}'*SigmaInv*designMatrix{i};
    PhiThetaMat(:, i) = designMatrix{i}*theta;
end

if(strcmp(theta_c.thetaPriorType, 'gaussian') || strcmp(theta_c.thetaPriorType, 'RVM'))
    %Find prior hyperparameter
    converged = false;
    iter = 0;
    while(~converged)
        if strcmp(theta_c.thetaPriorType, 'gaussian')
            stabilityParam = 1e-12;    %for stability in matrix inversion
            SigmaTilde = inv(sumPhiTSigmaInvPhi + (theta_c.priorHyperparam + stabilityParam)*I);
            muTilde = SigmaTilde*sumPhiTSigmaInvXmean;
            theta_prior_hyperparam_old = theta_c.priorHyperparam;
            theta_c.priorHyperparam = dim_theta/(muTilde'*muTilde + trace(SigmaTilde));
            if(abs(theta_c.priorHyperparam - theta_prior_hyperparam_old)/abs(theta_c.priorHyperparam) < 1e-5 || iter > 100)
                converged = true;
            elseif(~isfinite(theta_c.priorHyperparam) || theta_c.priorHyperparam <= 0)
                converged = true;
                theta_c.priorHyperparam = 1;
                warning('Gaussian hyperparameter precision is negative or not a number. Setting it to 1.')
            end
            
        elseif strcmp(theta_c.thetaPriorType, 'RVM')
            stabilityParam = 1e-12;    %for stability in matrix inversion
            SigmaTilde = inv(sumPhiTSigmaInvPhi + diag(theta_c.priorHyperparam + stabilityParam));
            muTilde = SigmaTilde*sumPhiTSigmaInvXmean;
            theta_prior_hyperparam_old = theta_c.priorHyperparam;
            theta_c.priorHyperparam = 1./(muTilde.^2 + diag(SigmaTilde));
            if(norm(theta_c.priorHyperparam - theta_prior_hyperparam_old)/norm(theta_c.priorHyperparam) < 1e-5 || iter > 100)
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
        sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + (theta_c.priorHyperparam(1) + stabilityParam)*I;
    elseif strcmp(theta_c.thetaPriorType, 'RVM')
        sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + diag(theta_c.priorHyperparam + stabilityParam);
    elseif strcmp(theta_c.thetaPriorType, 'none')
        
    else
        error('Unknown prior on theta_c')
    end
    
    if (strcmp(theta_c.thetaPriorType, 'gaussian') || strcmp(theta_c.thetaPriorType, 'RVM') ||...
            strcmp(theta_c.thetaPriorType, 'none'))
        theta_temp = sumPhiTSigmaInvPhi\sumPhiTSigmaInvXmean;
    else
        %         theta_temp = U*((sigma2*I + U*Phi.sumPhiTPhi*U)\U)*sumPhiTSigmaInvXmean;
        theta_temp = U*((U*sumPhiTSigmaInvPhi*U + I)\U)*sumPhiTSigmaInvXmean;
    end
    [~, msgid] = lastwarn;     %to catch nearly singular matrix
    
    
    gradHessTheta = @(theta) dF_dtheta(theta, theta_old, theta_c.thetaPriorType, theta_c.priorHyperparam, nTrain,...
        sumPhiTSigmaInvXmean, sumPhiTSigmaInvPhi);
    if(strcmp(msgid, 'MATLAB:singularMatrix') || strcmp(msgid, 'MATLAB:nearlySingularMatrix')...
            || strcmp(msgid, 'MATLAB:illConditionedMatrix') || norm(theta_temp)/length(theta) > 1e8)
        warning('theta_c is assuming unusually large values. Only go small step.')
        %         theta = fsolve(gradHessTheta, theta, fsolve_options_theta);
        theta = .5*(theta_old + .1*(norm(theta_old)/norm(theta_temp))*theta_temp)
        if any(~isfinite(theta))
            %restart from 0
            warning('Some components of theta are not finite. Restarting from theta = 0...')
            theta = 0*theta_old;
        end
        %Newton-Raphson maximization
        startValueTheta = theta;
        normGradientTol = eps;
        provide_objective = false;
        debugNRmax = false;
        RMMode = false;
        stepSizeTheta = .6;
        %             theta = newtonRaphsonMaximization(gradHessTheta, startValueTheta,...
        %                 normGradientTol, provide_objective, stepSizeTheta, RMMode, debugNRmax);
    else
        theta = theta_temp;
    end
    
    PhiThetaMat = zeros(nCoarse, nTrain);
    for i = 1:nTrain
        PhiThetaMat(:, i) = designMatrix{i}*theta;
    end
    
    %     theta = .5*theta + .5*theta_old;    %for stability
    
    if strcmp(sigma_prior_type, 'none')
        %         sigma2 = (1/(nTrain*nCoarse))*(sumXNormSqMean - 2*theta'*sumPhiTSigmaInvXmean + theta'*Phi.sumPhiTPhi*theta);
        Sigma = sparse(1:nCoarse, 1:nCoarse, mean(XSqMean - 2*(PhiThetaMat.*XMean) + PhiThetaMat.^2, 2));
        Sigma(Sigma < 0) = eps; %for numerical stability
        %         sigma2CutoffHi = 100;
        %         sigma2CutoffLo = 1e-50;
        %         if any(diag(Sigma) < sigma2CutoffLo)
        %             warning('sigma2 < cutoff. Set it to small cutoff value')
        %             %         sigma2 = sigma2CutoffLo;
        %             s = diag(Sigma);
        %             s(s < sigma2CutoffLo) = sigma2CutoffLo;
        %             index = 1:nCoarse;
        %             Sigma = sparse(index, index, s);
        %         elseif any(any(Sigma > sigma2CutoffHi))
        %             warning('sigma2 > cutoff, set it to cutoff')
        %             Sigma(Sigma > sigma2CutoffHi) = sigma2CutoffHi;
        %         end
        
        %sum_i Phi_i^T Sigma^-1 <X^i>_qi
        sumPhiTSigmaInvXmean = 0;
        %Only valid for diagonal Sigma
        s = diag(Sigma);
        SigmaInv = sparse(diag(1./s));
        SigmaInvXMean = SigmaInv*XMean;
        sumPhiTSigmaInvPhi = 0;
        
        for i = 1:nTrain
            sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + designMatrix{i}'*SigmaInvXMean(:, i);
            sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + designMatrix{i}'*SigmaInv*designMatrix{i};
        end
    elseif strcmp(sigma_prior_type, 'expSigSq')
        %Vector of variances
        sigmaSqVec = .5*(1./sigma_prior_hyperparam).*...
            (-.5*nTrain + sqrt(2*nTrain*sigma_prior_hyperparam.*...
            mean(XSqMean - 2*(PhiThetaMat.*XMean) + PhiThetaMat.^2, 2) + .25*nTrain^2));
        
        Sigma = sparse(1:nCoarse, 1:nCoarse, sigmaSqVec);
        Sigma(Sigma < 0) = eps; %for numerical stability
        sumPhiTSigmaInvXmean = 0;
        %Only valid for diagonal Sigma
        s = diag(Sigma);
        SigmaInv = sparse(diag(1./s));
        SigmaInvXMean = SigmaInv*XMean;
        sumPhiTSigmaInvPhi = 0;
        
        for i = 1:nTrain
            sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + designMatrix{i}'*SigmaInvXMean(:, i);
            sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + designMatrix{i}'*SigmaInv*designMatrix{i};
        end
%             error('Prior on diagonal Sigma not yet implemented')
%             gradHessLogSigmaMinus2 = @(lSigmaMinus2) dF_dlogSigmaMinus2(lSigmaMinus2, theta, nCoarse, nTrain, XNormSqMean,...
%                 sumPhiTSigmaInvXmean, Phi.sumPhiTPhi, sigma_prior_type, sigma_prior_hyperparam);
%             %     startValueLogSigmaMinus2 = logSigmaMinus2;
%             %     stepSizeSigma = .9; %the larger the faster, the smaller the more stable
%             %     logSigmaMinus2 = newtonRaphsonMaximization(gradHessLogSigmaMinus2, startValueLogSigmaMinus2,...
%             %         normGradientTol, provide_objective, stepSizeSigma, debugNRmax);
%             logSigmaMinus2 = fsolve(gradHessLogSigmaMinus2, logSigmaMinus2, fsolve_options_sigma);
%             sigmaMinus2 = exp(logSigmaMinus2);
%             sigma2_old = sigma2;
%             sigma2 = 1/sigmaMinus2;

    elseif strcmp(sigma_prior_type, 'delta')
        %Don't change sigma
        
        %sum_i Phi_i^T Sigma^-1 <X^i>_qi
        sumPhiTSigmaInvXmean = 0;
        %Only valid for diagonal Sigma
        s = diag(Sigma);
        SigmaInv = sparse(diag(1./s));
        SigmaInvXMean = SigmaInv*XMean;
        sumPhiTSigmaInvPhi = 0;
        
        for i = 1:nTrain
            sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + designMatrix{i}'*SigmaInvXMean(:, i);
            sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + designMatrix{i}'*SigmaInv*designMatrix{i};
        end
    end
    
    iter = iter + 1;
    thetaDiffRel = norm(theta_old_old - theta)/(norm(theta)*numel(theta));
    if((iter > 5 && thetaDiffRel < 1e-8) || iter > 200)
        converged = true;
    end
end

theta_c.theta = theta;
% theta_c.sigma = sqrt(sigma2);
theta_c.Sigma = Sigma;
theta_c.SigmaInv = SigmaInv;
theta_c.priorHyperparam = theta_c.priorHyperparam;
%Value which could be used to refine mesh
noPriorSigma = mean(XSqMean - 2*(PhiThetaMat.*XMean) + PhiThetaMat.^2, 2)
save('./data/noPriorSigma.mat', 'noPriorSigma');

end




