%% Main script for 2d coarse-graining
%% Preamble
tic;    %measure runtime
clear;
datestr(now, 'mmddHHMMSS')  %Print datestring to pipe
restoredefaultpath
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
addpath('./efficientVI')

rng('shuffle')  %system time seed

%initialize reduced order model object
romObj = ROM_SPDE

%% Load training data
romObj = romObj.loadTrainingData;
%Get model and training parameters
params;

%Open parallel pool
ppool = parPoolInit(romObj.nTrain);
pend = 0;       %for sequential qi-updates
%prealloc
XMean = zeros(romObj.coarseScaleDomain.nEl, romObj.nTrain);
XSqMean = ones(romObj.coarseScaleDomain.nEl, romObj.nTrain);

%% Compute design matrices
Phi = DesignMatrix(romObj.fineScaleDomain, romObj.coarseScaleDomain, romObj.featureFunctions,...
    romObj.globalFeatureFunctions, romObj.trainingDataMatfile, romObj.nStart:(romObj.nStart + romObj.nTrain - 1));
Phi = Phi.computeDesignMatrix(romObj.coarseScaleDomain.nEl, romObj.fineScaleDomain.nEl,...
    romObj.conductivityTransformation);
%Normalize design matrices
if romObj.standardizeFeatures
    Phi = Phi.standardizeDesignMatrix;
    Phi.saveNormalization('standardization');
elseif romObj.rescaleFeatures
    Phi = Phi.rescaleDesignMatrix;
    Phi.saveNormalization('rescaling');
end
%Compute sum_i Phi^T(x_i)^Phi(x_i)
if strcmp(romObj.mode, 'useNeighbor')
    %use feature function information from nearest neighbors
    Phi = Phi.includeNearestNeighborFeatures([romObj.coarseScaleDomain.nElX romObj.coarseScaleDomain.nElY]);
elseif strcmp(romObj.mode, 'useLocalNeighbor')
    Phi = Phi.includeLocalNearestNeighborFeatures([romObj.coarseScaleDomain.nElX romObj.coarseScaleDomain.nElY]);
elseif strcmp(romObj.mode, 'useLocalDiagNeighbor')
    Phi = Phi.includeLocalDiagNeighborFeatures([romObj.coarseScaleDomain.nElX romObj.coarseScaleDomain.nElY]);
elseif strcmp(romObj.mode, 'useDiagNeighbor')
    %use feature function information from nearest and diagonal neighbors
    Phi = Phi.includeDiagNeighborFeatures([romObj.coarseScaleDomain.nElX romObj.coarseScaleDomain.nElY]);
elseif strcmp(romObj.mode, 'useLocal')
    %Use separate parameters for every macro-cell
    Phi = Phi.localTheta_c([romObj.coarseScaleDomain.nElX romObj.coarseScaleDomain.nElY]);
end
Phi = Phi.computeSumPhiTPhi;
if strcmp(romObj.mode, 'useLocal')
    Phi.sumPhiTPhi = sparse(Phi.sumPhiTPhi);
end

MonteCarlo = false;
VI = true;

%% EM optimization - main body
k = 1;          %EM iteration index
collectData;    %Write initial parametrizations to disk

Xmax{1} = 0;
Xmax = repmat(Xmax, romObj.nTrain, 1);
epoch = 0;  %Number of times every data point has been seen
epoch_old = epoch;  %epoch of previous iteration
basisUpdated = 0;
while true
    k = k + 1;
    
    %% Establish distribution to sample from
    for i = 1:romObj.nTrain
        Tf_i_minus_mu = romObj.fineScaleDataOutput(:, i) - romObj.theta_cf.mu;
        PhiMat = Phi.designMatrices{i};
        %this transfers less data in parfor loops
        tcf = romObj.theta_cf;
        tcf.Sinv = [];
        tcf.sumLogS = sum(log(tcf.S));
        tcf.S = [];
        tc = romObj.theta_c;
        cd = romObj.coarseScaleDomain;
        ct = romObj.conductivityTransformation;
        log_qi{i} = @(Xi) log_q_i(Xi, Tf_i_minus_mu, tcf, tc,...
            PhiMat, cd, ct);
        premax = false;
        if(VI && premax)
            %This might be not worth the overhead, i.e. it is expensive
            if(k == 2 && ~loadOldConf)
                %Initialize VI distributions from maximum of q_i's
                Xmax{i} = max_qi(log_qi{i}, varDistParams{i}.mu');
                varDistParams{i}.mu = Xmax{i}';
            end
        end
    end
    clear PhiMat;
    
    

    if MonteCarlo
        for i = 1:romObj.nTrain
            %take MCMC initializations at mode of p_c
            MCMC(i).Xi_start = Phi.designMatrices{i}*romObj.theta_c.theta;
        end
        %% Test run for step sizes
        disp('test sampling...')
        parfor i = 1:romObj.nTrain
            Tf_i_minus_mu = romObj.fineScaleDataOutput(:, i) - romObj.theta_cf.mu;
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
                    MCMCstepWidth(i).MALA.stepWidth = (1/.7)*(outStepWidth(i).acceptance +...
                        (1 - outStepWidth(i).acceptance)*.1)*...
                        MCMCstepWidth(i).MALA.stepWidth;
                elseif strcmp(MCMCstepWidth(i).method, 'randomWalk')
                    MCMCstepWidth(i).randomWalk.proposalCov = (1/.7)*(outStepWidth(i).acceptance +...
                        (1 - outStepWidth(i).acceptance)*.1)*MCMCstepWidth(i).randomWalk.proposalCov;
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
        
        for i = 1:romObj.nTrain
            if(k - 1 <= length(nSamplesBeginning))
                %less samples at the beginning
                MCMC(i).nSamples = nSamplesBeginning(k - 1);
            end
        end
        
        disp('actual sampling...')
        %% Generate samples from every q_i
        parfor i = 1:romObj.nTrain
            Tf_i_minus_mu = romObj.fineScaleDataOutput(:, i) - romObj.theta_cf.mu;
            %sample from every q_i
            out(i) = MCMCsampler(log_qi{i}, Xmax{i}, MCMC(i));
            %avoid very low acceptances
            while out(i).acceptance < .1
                out(i) = MCMCsampler(log_qi{i}, Xmax{i}, MCMC(i));
                %if there is a second loop iteration, take last sample as initial position
                MCMC(i).Xi_start = out(i).samples(:,end);
                if strcmp(MCMC(i).method, 'MALA')
                    MCMC(i).MALA.stepWidth = (1/.9)*(out(i).acceptance +...
                        (1 - out(i).acceptance)*.1)*MCMC(i).MALA.stepWidth;
                elseif strcmp(MCMC(i).method, 'randomWalk')
                    MCMC(i).randomWalk.proposalCov = .2*MCMC(i).randomWalk.proposalCov;
                    MCMC(i).randomWalk.proposalCov = (1/.7)*(out(i).acceptance + (1 -...
                        out(i).acceptance)*.1)*MCMC(i).randomWalk.proposalCov;
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
            
            %for S
            %Tc_samples(:,:,i) contains coarse nodal temperature samples (1 sample == 1 column) for full order data
            %sample i
            Tc_samples(:, :, i) = reshape(cell2mat(out(i).data), romObj.coarseScaleDomain.nNodes, MCMC(i).nSamples);
            %only valid for diagonal S here!
            varExpect_p_cf_exp(:, i) = mean((repmat(Tf_i_minus_mu, 1, MCMC(i).nSamples)...
                - romObj.theta_cf.W*Tc_samples(:, :, i)).^2, 2);
            
        end
        clear Tc_samples;
    elseif VI
        
        if (strcmp(update_qi, 'sequential') && k > 2)
            %Sequentially update N_threads qi's at a time, then perform M-step
            epoch_old = epoch;
            pstart = pend + 1;
            if pstart > romObj.nTrain
                pstart = 1;
                epoch = epoch + 1;
            end
            pend = pstart + ppool.NumWorkers - 1;
            if pend > romObj.nTrain
                pend = romObj.nTrain;
            elseif pend < pstart
                pend = pstart;
            end
        else
            pstart = 1;
            pend = romObj.nTrain;
        end
        
        %This can probably be done more memory efficient
        disp('Finding optimal variational distributions...')
%         ticBytes(gcp)
%         parfor i = pstart:pend
%             so{i}.gradientHandle = @(x) vi{i}.gradientHandle(x);
%             so{i} = so{i}.converge;
%         end
%         tocBytes(gcp)

        dim = romObj.coarseScaleDomain.nEl;
        tic
        ticBytes(gcp)
        parfor i = pstart:pend
            [varDistParams{i}, varDistParamsVec{i}] = efficientStochOpt(varDistParamsVec{i},...
                log_qi{i}, variationalDist, sw, dim);
        end
        tocBytes(gcp)
        parfor_time = toc
        
        %Sample from VI distributions and solve coarse model
%         for i = pstart:pend
%             vi{i} = vi{i}.setVarDistParams(vi{i}.params_vec2struc(so{i}.x));
%             
%             XMean(:, i) = vi{i}.moments{1}';
%             XSqMean(:, i) = diag(vi{i}.moments{2});
%             
%             Tf_i_minus_mu = romObj.fineScaleDataOutput(:, i) - romObj.theta_cf.mu;
%             p_cf_expHandle{i} = @(logCond) p_cf_expfun(logCond, romObj.conductivityTransformation,...
%                 romObj.coarseScaleDomain, Tf_i_minus_mu, romObj.theta_cf);
%             %Expectations under variational distributions
%             varExpect_p_cf_exp(:, i) = vi{i}.mcInference(p_cf_expHandle{i});
%         end
        
        tic
        for i = pstart:pend            
            XMean(:, i) = varDistParams{i}.mu';
            XSqMean(:, i) = varDistParams{i}.XSqMean;
            
            Tf_i_minus_mu = romObj.fineScaleDataOutput(:, i) - romObj.theta_cf.mu;
            p_cf_expHandle{i} = @(logCond) p_cf_expfun(logCond, romObj.conductivityTransformation,...
                romObj.coarseScaleDomain, Tf_i_minus_mu, romObj.theta_cf);
            %Expectations under variational distributions
            varExpect_p_cf_exp(:, i) = mcInference(p_cf_expHandle{i}, variationalDist, varDistParams{i});
        end
        inference_time = toc
        tic
%         save('./data/variationalDistributions.mat', 'vi');
        save_time = toc
        disp('done')
        
        
        
        
    end
    
    %% M-step: determine optimal parameters given the sample set
    disp('M-step: find optimal params')
    %Optimal S (decelerated convergence)
    lowerBoundS = eps;
    romObj.theta_cf.S = (1 - mix_S)*mean(varExpect_p_cf_exp, 2)...
        + mix_S*romObj.theta_cf.S + lowerBoundS*ones(romObj.fineScaleDomain.nNodes, 1);
    romObj.theta_cf.Sinv = sparse(1:romObj.fineScaleDomain.nNodes,...
        1:romObj.fineScaleDomain.nNodes, 1./romObj.theta_cf.S);
    romObj.theta_cf.Sinv_vec = 1./romObj.theta_cf.S;
    romObj.theta_cf.WTSinv = romObj.theta_cf.W'*romObj.theta_cf.Sinv;        %Precomputation for efficiency

    %optimal theta_c and sigma
    Sigma_old = romObj.theta_c.Sigma;
    theta_old = romObj.theta_c.theta;
%     %Adjust hyperprior to not get stuck at 0
%     if(k - 1 <= size(theta_prior_hyperparamArray, 1))
%         theta_prior_hyperparam = theta_prior_hyperparamArray(k - 1, :)
%     end
    
    if(k > fixSigInit)
        sigma_prior_type = sigma_prior_type_hold;
    else
        sigma_prior_type = 'delta';
    end
    
    romObj.theta_c = optTheta_c(romObj.theta_c, romObj.nTrain, romObj.coarseScaleDomain.nEl, XSqMean,...
        Phi, XMean, sigma_prior_type, sigma_prior_hyperparam);
    romObj.theta_c.Sigma = (1 - mix_sigma)*romObj.theta_c.Sigma + mix_sigma*Sigma_old;
    
    k
    if(~romObj.theta_c.useNeuralNet)
        romObj.theta_c.theta = (1 - mix_theta)*romObj.theta_c.theta + mix_theta*theta_old;
        disp('M-step done, current params:')
        [~, index] = sort(abs(romObj.theta_c.theta));
        if strcmp(romObj.mode, 'useNeighbor')
            feature = mod((index - 1), numel(Phi.featureFunctions)) + 1;
            %counted counterclockwise from right to lower neighbor
            neighborElement = floor((index - 1)/numel(Phi.featureFunctions));
            curr_theta = [romObj.theta_c.theta(index) feature neighborElement]
        elseif strcmp(romObj.mode, 'useDiagNeighbor')
            feature = mod((index - 1), numel(Phi.featureFunctions)) + 1;
            %counted counterclockwise from right to lower right neighbor
            neighborElement = floor((index - 1)/numel(Phi.featureFunctions));
            curr_theta = [romObj.theta_c.theta(index) feature neighborElement]
        elseif strcmp(romObj.mode, 'useLocal')
            feature = mod((index - 1), size(Phi.featureFunctions, 2) + size(Phi.globalFeatureFunctions, 2)) + 1;
            Element = floor((index - 1)/(size(Phi.featureFunctions, 2) + size(Phi.globalFeatureFunctions, 2))) + 1;
            curr_theta = [romObj.theta_c.theta(index) feature Element]
        elseif(strcmp(romObj.mode, 'useLocalNeighbor') || strcmp(romObj.mode, 'useLocalDiagNeighbor'))
            disp('theta feature coarseElement neighbor')
            curr_theta = [romObj.theta_c.theta(index) Phi.neighborDictionary(index, 1)...
                Phi.neighborDictionary(index, 2) Phi.neighborDictionary(index, 3)]
        else
            curr_theta = [romObj.theta_c.theta(index) index]
        end
        curr_theta_hyperparam = romObj.theta_c.priorHyperparam
        
        if(romObj.linFiltSeq && epoch > initialEpochs && mod((epoch - initialEpochs + 1), basisUpdateGap) == 0 &&...
                epoch ~= epoch_old && basisUpdated < basisFunctionUpdates)
            basisUpdated = basisUpdated + 1;
%             [romObj, Phi] = romObj.addLinearFilterFeature(XMean, Phi);
            [romObj, Phi] = romObj.addGlobalLinearFilterFeature(XMean, Phi);
            %Recompute theta_c and sigma
            romObj.theta_c = optTheta_c(romObj.theta_c, romObj.nTrain, romObj.coarseScaleDomain.nEl, XSqMean,...
                Phi, XMean, sigma_prior_type, sigma_prior_hyperparam);
            romObj.theta_c.Sigma = (1 - mix_sigma)*romObj.theta_c.Sigma + mix_sigma*Sigma_old;
        end
        
        plotTheta = false;
        if plotTheta
            if ~exist('thetaArray')
                thetaArray = romObj.theta_c.theta';
            else
                if(size(romObj.theta_c.theta, 1) > size(thetaArray, 2))
                    %New basis function included. Expand array
                    thetaArray = [thetaArray, zeros(size(thetaArray, 1),...
                        numel(romObj.theta_c.theta) - size(thetaArray, 2))];
                    thetaArray = [thetaArray; romObj.theta_c.theta'];
                else
                    thetaArray = [thetaArray; romObj.theta_c.theta'];
                end
            end
            if ~exist('thetaHyperparamArray')
                thetaHyperparamArray = romObj.theta_c.priorHyperparam';
            else
                if(size(romObj.theta_c.priorHyperparam, 1) > size(thetaHyperparamArray, 2))
                    %New basis function included. Expand array
                    thetaHyperparamArray = [thetaHyperparamArray, zeros(size(thetaHyperparamArray, 1),...
                        numel(romObj.theta_c.priorHyperparam) - size(thetaHyperparamArray, 2))];
                    thetaHyperparamArray = [thetaHyperparamArray; romObj.theta_c.priorHyperparam'];
                else
                    thetaHyperparamArray = [thetaHyperparamArray; romObj.theta_c.priorHyperparam'];
                end
            end
            if ~exist('sigmaArray')
                sigmaArray = diag(romObj.theta_c.Sigma)';
            else
                sigmaArray = [sigmaArray; diag(romObj.theta_c.Sigma)'];
            end
            if exist('figureTheta')
                figure(figureTheta);
            else
                figureTheta = figure;
            end
            subplot(3,2,1)
            plot(thetaArray, 'linewidth', 1)
            axis tight;
            subplot(3,2,2)
            plot(romObj.theta_c.theta, 'linewidth', 1)
            axis tight;
            subplot(3,2,3)
            semilogy(sqrt(sigmaArray), 'linewidth', 1)
            axis tight;
            subplot(3,2,4)
            imagesc(reshape(diag(sqrt(romObj.theta_c.Sigma)), romObj.coarseScaleDomain.nElX,...
                romObj.coarseScaleDomain.nElY))
            title('\sigma_k')
            colorbar
            grid off;
            axis tight;
            subplot(3,2,5)
            plot(thetaHyperparamArray, 'linewidth', 1)
            axis tight;
            subplot(3,2,6)
            plot(romObj.theta_c.priorHyperparam, 'linewidth', 1)
            axis tight;
            drawnow
        end
    end
    
    curr_sigma = romObj.theta_c.Sigma
    mean_S = mean(romObj.theta_cf.S)
    if(~romObj.conductivityTransformation.anisotropy)
        if romObj.theta_c.useNeuralNet
            m = predict(romObj.theta_c.theta, xkNN(:, :, 1, 1:romObj.coarseScaleDomain.nEl));
            Lambda_eff1_mode = conductivityBackTransform(m, romObj.conductivityTransformation)
        else
            Lambda_eff1_mode = conductivityBackTransform(Phi.designMatrices{1}*romObj.theta_c.theta,...
                romObj.conductivityTransformation)
        end
    end
    
    %collect data and write it to disk periodically to save memory
    collectData;
    if(epoch > maxEpochs)
        break;
    end
end
%tidy up
clear i j k m Wa Wa_mean Tc_dyadic_mean log_qi p_cf_exponent;
runtime = toc

































