%% Main script for 2d coarse-graining
%% Preamble
tic;    %measure runtime
clear all;
datestr(now, 'mmddHHMMSS')  %Print datestring to pipe

addpath('./params')
addpath('./aux')
addpath('./heatFEM')
addpath('./rom')
addpath('./computation')
addpath('./FEMgradient')
addpath('./MCMCsampler')
addpath('./optimization')
addpath('./genConductivity')
addpath('./variationalInference')
addpath('./featureFunctions')

rng('shuffle')  %system time seed

%% Load training data
loadTrainingData;
%Get model and training parameters
params;

%Open parallel pool
parPoolInit(nTrain);
ppool = gcp;    %parallel pool properties
pend = 0;       %for sequential qi-updates
%initializations
XMean = zeros(domainc.nEl, nTrain);
XNormSqMean = zeros(1, nTrain);

Tf = Tffile.Tf(:, nStart:(nStart + nTrain - 1));        %Finescale temperatures - load partially to save memory

%% Compute design matrices
Phi = DesignMatrix([domainf.nElX domainf.nElY], [domainc.nElX domainc.nElY], phi, Tffile, nStart:(nStart + nTrain - 1));
Phi = Phi.computeDesignMatrix(domainc.nEl, domainf.nEl);
%Normalize design matrices
Phi = Phi.computeFeatureFunctionAbsMean;
Phi = Phi.normalizeDesignMatrix;
Phi.saveNormalization;
%Compute sum_i Phi^T(x_i)^Phi(x_i)
Phi = Phi.computeSumPhiTPhi;


%% EM optimization - main body
k = 1;          %EM iteration index
collectData;    %Write initial parametrizations to disk

for k = 2:(maxIterations + 1)

    %% Establish distribution to sample from
    for i = 1:nTrain
        Tf_i_minus_mu = Tf(:, i) - theta_cf.mu;
        log_qi{i} = @(Xi) log_q_i(Xi, Tf_i_minus_mu, theta_cf, theta_c,...
            Phi.designMatrices{i}, domainc);
    end
    
    MonteCarlo = false;
    VI = true;

    if MonteCarlo
        for i = 1:nTrain
            %take MCMC initializations at mode of p_c
            MCMC(i).Xi_start = Phi.designMatrices{i}*theta_c.theta;
        end
        %% Test run for step sizes
        disp('test sampling...')
        parfor i = 1:nTrain
            Tf_i_minus_mu = Tf(:, i) - theta_cf.mu;
            %find maximum of qi for thermalization
            %start value has some randomness to drive transitions between local optima
            X_start{i} = normrnd(MCMC(i).Xi_start, .01);
            Xmax{i} = max_qi(log_qi{i}, X_start{i});
            
            %sample from every q_i
            outStepWidth(i) = MCMCsampler(log_qi{i}, Xmax{i}, MCMCstepWidth(i));
            while (outStepWidth(i).acceptance < .5 || outStepWidth(i).acceptance > .9)
                outStepWidth(i) = MCMCsampler(log_qi{i}, Xmax{i}, MCMCstepWidth(i));
                MCMCstepWidth(i).Xi_start = outStepWidth(i).samples(:, end);
                if strcmp(MCMCstepWidth(i).method, 'MALA')
                    MCMCstepWidth(i).MALA.stepWidth = (1/.7)*(outStepWidth(i).acceptance + (1 - outStepWidth(i).acceptance)*.1)*...
                        MCMCstepWidth(i).MALA.stepWidth;
                elseif strcmp(MCMCstepWidth(i).method, 'randomWalk')
                    MCMCstepWidth(i).randomWalk.proposalCov = (1/.7)*(outStepWidth(i).acceptance + (1 - outStepWidth(i).acceptance)*.1)*MCMCstepWidth(i).randomWalk.proposalCov;
                else
                    error('Unknown MCMC method')
                end
            end
            %Set step widths and start values
            if strcmp(MCMCstepWidth(i).method, 'MALA')
                MCMC(i).MALA.stepWidth = MCMCstepWidth(i).MALA.stepWidth;
            elseif strcmp(MCMCstepWidth(i).method, 'randomWalk')
                MCMC(i).randomWalk.proposalCov = MCMCstepWidth(i).randomWalk.proposalCov;
            else
                error('Unknown MCMC method')
            end
            MCMC(i).Xi_start = MCMCstepWidth(i).Xi_start;
        end
        
        for i = 1:nTrain
            if(k - 1 <= length(nSamplesBeginning))
                %less samples at the beginning
                MCMC(i).nSamples = nSamplesBeginning(k - 1);
            end
        end
        
        disp('actual sampling...')
        %% Generate samples from every q_i
        parfor i = 1:nTrain
            Tf_i_minus_mu = Tf(:, i) - theta_cf.mu;
            %sample from every q_i
            out(i) = MCMCsampler(log_qi{i}, Xmax{i}, MCMC(i));
            %avoid very low acceptances
            while out(i).acceptance < .1
                out(i) = MCMCsampler(log_qi{i}, Xmax{i}, MCMC(i));
                %if there is a second loop iteration, take last sample as initial position
                MCMC(i).Xi_start = out(i).samples(:,end);
                if strcmp(MCMC(i).method, 'MALA')
                    MCMC(i).MALA.stepWidth = (1/.9)*(out(i).acceptance + (1 - out(i).acceptance)*.1)*MCMC(i).MALA.stepWidth;
                elseif strcmp(MCMC(i).method, 'randomWalk')
                    MCMC(i).randomWalk.proposalCov = .2*MCMC(i).randomWalk.proposalCov;
                    MCMC(i).randomWalk.proposalCov = (1/.7)*(out(i).acceptance + (1 - out(i).acceptance)*.1)*MCMC(i).randomWalk.proposalCov;
                else
                    error('Unknown MCMC method')
                end
                warning('Acceptance ratio below .1')
            end
            
            %Refine step width
            if strcmp(MCMC(i).method, 'MALA')
                MCMC(i).MALA.stepWidth = (1/.7)*out(i).acceptance*MCMC(i).MALA.stepWidth;
            elseif strcmp(MCMC(i).method, 'randomWalk')
                MCMC(i).randomWalk.proposalCov = (1/.7)*out(i).acceptance*MCMC(i).randomWalk.proposalCov;
            else
            end
            
            XMean(:, i) = mean(out(i).samples, 2);
            XNormSqMean(i) = mean(sum(out(i).samples.^2));
            
            %for S
            %Tc_samples(:,:,i) contains coarse nodal temperature samples (1 sample == 1 column) for full order data
            %sample i
            Tc_samples(:, :, i) = reshape(cell2mat(out(i).data), domainc.nNodes, MCMC(i).nSamples);
            %only valid for diagonal S here!
            p_cf_exponent(:, i) = mean((repmat(Tf_i_minus_mu, 1, MCMC(i).nSamples) - theta_cf.W*Tc_samples(:, :, i)).^2, 2);
            
        end
        clear Tc_samples;
    elseif VI
        
        if strcmp(update_qi, 'sequential')
            %Sequentially update N_threads qi's at a time, then perform M-step
            pstart = pend + 1;
            if pstart > nTrain
                pstart = 1;
            end
            pend = pstart + ppool.NumWorkers - 1;
            if pend > nTrain
                pend = nTrain;
            end
        else
            pstart = 1;
            pend = nTrain;
        end
        
        disp('Finding optimal variational distributions...')
        parfor i = pstart:pend
            [optVarDist{i}, RMsteps{i}] = variationalInference(log_qi{i}, VIparams, initialParamsArray{i});
            initialParamsArray{i} = optVarDist{i}.params;
            if(strcmp(VIparams.family, 'diagonalGaussian'))
                XMean(:, i) = optVarDist{i}.params(1:domainc.nEl);
                XNormSqMean(i) = sum([optVarDist{i}.params(1:domainc.nEl).^2 exp(-optVarDist{i}.params((domainc.nEl + 1):end))]);
            else
                error('VI not implemented for this family of functions')
            end
        end
        %Set start values for next iteration
        for i = pstart:pend
            VIparams.initialParams{i} = optVarDist{i}.params;
        end
        disp('done')
        %Sample from VI distributions and solve coarse model
        for i = pstart:pend
            Tf_i_minus_mu = Tf(:, i) - theta_cf.mu;
            if(strcmp(VIparams.family, 'diagonalGaussian'))
                VImean = optVarDist{i}.params(1:domainc.nEl);
                VIsigma = exp(-.5*optVarDist{i}.params((domainc.nEl + 1):end));
                %Samples of conductivity
                samples = logCond2Cond(mvnrnd(VImean, VIsigma, VIparams.inferenceSamples), 1e-10, 1e10);
                
                for s = 1:VIparams.inferenceSamples
                    for j = 1:domainc.nEl
                        D(:, :, j) =  samples(s, j)*eye(2);
                    end
                    FEMout = heat2d(domainc, D);
                    
                    Tc = FEMout.Tff';
                    Tc_samples(:, s, i) = Tc(:);
                end
                p_cf_exponent(:, i) = mean((repmat(Tf_i_minus_mu, 1, VIparams.inferenceSamples)...
                        - theta_cf.W*Tc_samples(:, :, i)).^2, 2);
            else
                error('VI not implemented for this family of functions')
            end
        end
        
        
        
        
    end
    
    %% M-step: determine optimal parameters given the sample set
    disp('M-step: find optimal params')
    %Optimal S (decelerated convergence)
    lowerBoundS = 1e-10;
    theta_cf.S = (1 - mix_S)*mean(p_cf_exponent, 2)...
        + mix_S*theta_cf.S + lowerBoundS*ones(domainf.nNodes, 1);
    clear p_cf_exponent;
    theta_cf.Sinv = sparse(1:domainf.nNodes, 1:domainf.nNodes, 1./theta_cf.S);
    theta_cf.WTSinv = theta_cf.W'*theta_cf.Sinv;        %Precomputation for efficiency

    %optimal theta_c and sigma
    %sum_i Phi_i^T <X^i>_qi
    sumPhiTXmean = zeros(numel(phi), 1);
    for i = 1:nTrain
        sumPhiTXmean = sumPhiTXmean + Phi.designMatrices{i}'*XMean(:, i);
    end

    sigma_old = theta_c.sigma;
    theta_c = optTheta_c(theta_c, nTrain, domainc.nEl, XNormSqMean,...
        sumPhiTXmean, Phi.sumPhiTPhi, theta_prior_type, theta_prior_hyperparam,...
        sigma_prior_type, sigma_prior_hyperparam);
    theta_c.sigma = (1 - mix_sigma)*theta_c.sigma + mix_sigma*sigma_old;
    disp('M-step done, current params:')
    k
    curr_theta = [theta_c.theta (1:length(theta_c.theta))']
    curr_sigma = theta_c.sigma
    mean_S = mean(theta_cf.S)
    Lambda_eff1_mode = exp(Phi.designMatrices{1}*theta_c.theta)

    %collect data and write it to disk periodically to save memory
    collectData;
end
%tidy up
clear i j k m Wa Wa_mean Tc_dyadic_mean log_qi p_cf_exponent curr_theta XMean XNormSqMean;
runtime = toc

































