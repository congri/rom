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
addpath('./autoencoder')

rng('shuffle')  %system time seed

delete('./data/*')  %delete old data

%initialize reduced order model object
rom = ROM_SPDE('train');
%prealloc for p_cf inference
tempArray = zeros(rom.fineMesh.nNodes, rom.nTrain);
%% Load training data
% romObj = romObj.loadTrainingData;
%Get model and training parameters
params;

%Open parallel pool
ppool = parPoolInit(rom.nTrain);
pend = 0;       %for sequential qi-updates

%Compute design matrices
rom.computeDesignMatrix('train', false);
if(size(rom.designMatrix{1}, 2) ~= size(rom.theta_c.theta))
    warning('Wrong dim of theta_c. Setting it to 0 with correct dim.')
    rom.theta_c.theta = zeros(size(rom.designMatrix{1}, 2), 1);
end
%random initialization
%romObj.theta_c.theta = normrnd(0, 1, size(romObj.theta_c.theta));
% romObj.theta_c.theta(1) = 0;

if strcmp(rom.inferenceMethod, 'monteCarlo')
    MonteCarlo = true;
else
    MonteCarlo = false;
end

%% EM optimization - main body
rom.EM_iterations = 1;          %EM iteration index
collectData;    %Write initial parametrizations to disk

Xmax{1} = 0;
Xmax = repmat(Xmax, rom.nTrain, 1);
while true
    rom.EM_iterations = rom.EM_iterations + 1;
    
    %% Establish distribution to sample from
    for i = 1:rom.nTrain
        Tf_i_minus_mu = rom.fineScaleDataOutput(:, i) - rom.theta_cf.mu;
        PhiMat = rom.designMatrix{i};
        %this transfers less data in parfor loops
        tcf = rom.theta_cf;
        tcf.Sinv = [];
        tcf.sumLogS = sum(log(tcf.S));
        tcf.S = [];
        tc = rom.theta_c;
        ct = rom.conductivityTransformation;
        
        if(any(rom.boundaryConditionVariance))
            %Set coarse domain for data with different boundary conditions
            nX = rom.coarseMesh.nElX;
            nY = rom.coarseMesh.nElY;
            bc = rom.trainingDataMatfile.bc;
            j = rom.trainingSamples(i);
            bcT = @(x)...
                bc{j}(1) + bc{j}(2)*x(1) + bc{j}(3)*x(2) + bc{j}(4)*x(1)*x(2);
            bcQ{1} = @(x) -(bc{j}(3) + bc{j}(4)*x);      %lower bound
            bcQ{2} = @(y) (bc{j}(2) + bc{j}(4)*y);       %right bound
            bcQ{3} = @(x) (bc{j}(3) + bc{j}(4)*x);       %upper bound
            bcQ{4} = @(y) -(bc{j}(2) + bc{j}(4)*y);      %left bound
            cm(i) = rom.coarseMesh;
            cm(i) = cm(i).setBoundaries([2:(2*nX + 2*nY)], bcT, bcQ);
            cd_i = cm(i);
            log_qi{i} = @(Xi)...
                log_q_i(Xi, Tf_i_minus_mu, tcf, tc, PhiMat, cd_i, ct, true);
            log_qi_max{i} = @(Xi)...
                log_q_i(Xi, Tf_i_minus_mu, tcf, tc, PhiMat, cd_i, ct, false);
        else
            %Every coarse model has the same boundary conditions
            cm = rom.coarseMesh;
            log_qi{i} = @(Xi)...
                log_q_i(Xi, Tf_i_minus_mu, tcf, tc, PhiMat, cm, ct, true);
            log_qi_max{i} = @(Xi)...
                log_q_i(Xi, Tf_i_minus_mu, tcf, tc, PhiMat, cm, ct, false);
        end
        
        
        premax = false;
        if(strcmp(rom.inferenceMethod, 'variationalInference') && premax)
            %This might be not worth the overhead, i.e. it is expensive
            if(rom.EM_iterations == 2 && ~loadOldConf)
                %Initialize VI distributions from maximum of q_i's
                Xmax{i} = max_qi(log_qi_max{i}, varDistParams{i}.mu');
                varDistParams{i}.mu = Xmax{i}';
            end
        end
    end
    clear PhiMat;
    
    
    
    if MonteCarlo
        for i = 1:rom.nTrain
            %take MCMC initializations at mode of p_c
            MCMC(i).Xi_start = rom.designMatrix{i}*rom.theta_c.theta;
        end
        %% Test run for step sizes
        disp('test sampling...')
        parfor i = 1:rom.nTrain
            %find maximum of qi for thermalization
            %start value has some randomness to drive transitions
            %between local optima
            X_start{i} = normrnd(MCMC(i).Xi_start, .01);
            Xmax{i} = max_qi(log_qi_max{i}, X_start{i});
            
            %sample from every q_i
            outStepWidth(i) = MCMCsampler(log_qi{i}, Xmax{i}, MCMCstepWidth(i));
            while (outStepWidth(i).acceptance < .5 ||...
                    outStepWidth(i).acceptance > .9)
                outStepWidth(i) =...
                    MCMCsampler(log_qi{i}, Xmax{i}, MCMCstepWidth(i));
                MCMCstepWidth(i).Xi_start = outStepWidth(i).samples(:, end);
                if strcmp(MCMCstepWidth(i).method, 'MALA')
                    MCMCstepWidth(i).MALA.stepWidth = ...
                        (1/.7)*(outStepWidth(i).acceptance +...
                        (1 - outStepWidth(i).acceptance)*.1)*...
                        MCMCstepWidth(i).MALA.stepWidth;
                elseif strcmp(MCMCstepWidth(i).method, 'randomWalk')
                    MCMCstepWidth(i).randomWalk.proposalCov =...
                        (1/.7)*(outStepWidth(i).acceptance +...
                        (1 - outStepWidth(i).acceptance)*.1)...
                        *MCMCstepWidth(i).randomWalk.proposalCov;
                else
                    error('Unknown MCMC method')
                end
            end
            %Set step widths and start values
            if strcmp(MCMCstepWidth(i).method, 'MALA')
                MCMC(i).MALA.stepWidth = MCMCstepWidth(i).MALA.stepWidth;
            elseif strcmp(MCMCstepWidth(i).method, 'randomWalk')
                MCMC(i).randomWalk.proposalCov =...
                    MCMCstepWidth(i).randomWalk.proposalCov;
            else
                error('Unknown MCMC method')
            end
            MCMC(i).Xi_start = MCMCstepWidth(i).Xi_start;
        end
        
        for i = 1:rom.nTrain
            if(rom.EM_iterations - 1 <= length(nSamplesBeginning))
                %less samples at the beginning
                MCMC(i).nSamples = nSamplesBeginning(rom.EM_iterations - 1);
            end
        end
        
        disp('actual sampling...')
        %% Generate samples from every q_i
        parfor i = 1:rom.nTrain
            Tf_i_minus_mu = rom.fineScaleDataOutput(:, i) - rom.theta_cf.mu;
            %sample from every q_i
            out(i) = MCMCsampler(log_qi{i}, Xmax{i}, MCMC(i));
            %avoid very low acceptances
            while out(i).acceptance < .1
                out(i) = MCMCsampler(log_qi{i}, Xmax{i}, MCMC(i));
                %if there is a second loop iteration, 
                %take last sample as initial position
                MCMC(i).Xi_start = out(i).samples(:,end);
                if strcmp(MCMC(i).method, 'MALA')
                    MCMC(i).MALA.stepWidth = (1/.9)*(out(i).acceptance +...
                        (1 - out(i).acceptance)*.1)*MCMC(i).MALA.stepWidth;
                elseif strcmp(MCMC(i).method, 'randomWalk')
                    MCMC(i).randomWalk.proposalCov =...
                        .2*MCMC(i).randomWalk.proposalCov;
                    MCMC(i).randomWalk.proposalCov =...
                        (1/.7)*(out(i).acceptance + (1 -...
                        out(i).acceptance)*.1)*MCMC(i).randomWalk.proposalCov;
                else
                    error('Unknown MCMC method')
                end
                warning('Acceptance ratio below .1')
            end
            
            %Refine step width
            if strcmp(MCMC(i).method, 'MALA')
                MCMC(i).MALA.stepWidth =...
                    (1/.7)*out(i).acceptance*MCMC(i).MALA.stepWidth;
            elseif strcmp(MCMC(i).method, 'randomWalk')
                MCMC(i).randomWalk.proposalCov = ...
                    (1/.7)*out(i).acceptance*MCMC(i).randomWalk.proposalCov;
            else
            end
            
            rom.XMean(:, i) = mean(out(i).samples, 2);
            
            %for S
            %Tc_samples(:,:,i) contains coarse nodal temperature samples
            %(1 sample == 1 column) for full order data
            %sample i
            Tc_samples(:, :, i) = reshape(cell2mat(out(i).data),...
                rom.coarseMesh.nNodes, MCMC(i).nSamples);
            %only valid for diagonal S here!
            tempArray(:, i)= mean((repmat(Tf_i_minus_mu, 1, MCMC(i).nSamples)...
                - rom.theta_cf.W*Tc_samples(:, :, i)).^2, 2);
            
        end
        clear Tc_samples;
    elseif strcmp(rom.inferenceMethod, 'variationalInference')
        
        if (strcmp(update_qi, 'sequential') && rom.EM_iterations > 2)
            %Sequentially update N_threads qi's at a time, then perform M-step
            rom.epoch_old = rom.epoch;
            pstart = pend + 1;
            if pstart > rom.nTrain
                pstart = 1;
                rom.epoch = rom.epoch + 1;
            end
            pend = pstart + 8*ppool.NumWorkers - 1;
            if pend > rom.nTrain
                pend = rom.nTrain;
            elseif pend < pstart
                pend = pstart;
            end
        else
            pstart = 1;
            pend = rom.nTrain;
        end
        
        %This can probably be done more memory efficient
        disp('Finding optimal variational distributions...')
        
        dim = rom.coarseMesh.nEl;
        tic
        ticBytes(gcp)
        for i = pstart:pend
            [varDistParams{i}, varDistParamsVec{i}] = efficientStochOpt(...
                varDistParamsVec{i}, log_qi{i}, variationalDist, sw, dim);
        end
        tocBytes(gcp)
        parfor_time = toc
        
        tic
        for i = pstart:pend
            rom.XMean(:, i) = varDistParams{i}.mu';
            rom.XSqMean(:, i) = varDistParams{i}.XSqMean;
            
            Tf_i_minus_mu = rom.fineScaleDataOutput(:, i) - rom.theta_cf.mu;
            if(any(rom.boundaryConditionVariance))
                p_cf_expHandle{i} = @(X) sqMisfit(X,...
                    rom.conductivityTransformation,...
                    cm(i), Tf_i_minus_mu, rom.theta_cf);
            else
                p_cf_expHandle{i} = @(X) sqMisfit(X,...
                    rom.conductivityTransformation,...
                    cm, Tf_i_minus_mu, rom.theta_cf);
            end
            %Expectations under variational distributions
            if rom.free_W
                [p_cf_exp, Tc_i, TcTcT_i] = mcInference(p_cf_expHandle{i},...
                    variationalDist, varDistParams{i});
                Tc(:, i) = Tc_i;
                TcTcT(:, :, i) = TcTcT_i;
            else
                p_cf_exp = mcInference(...
                    p_cf_expHandle{i}, variationalDist, varDistParams{i});
            end
            tempArray(:, i) = p_cf_exp;
        end
        if rom.free_W
            rom.mean_TfTcT = 0;
            for i = 1:rom.nTrain
                rom.mean_TfTcT = (1/i)*((i - 1)*rom.mean_TfTcT +...
                    rom.fineScaleDataOutput(:, i)*Tc(:, i)');
            end
            rom.mean_TcTcT = mean(TcTcT, 3);
        end
        rom.varExpect_p_cf_exp_mean = mean(tempArray, 2);
        inference_time = toc
        tic
        %save('./data/variationalDistributions.mat', 'vi');
        save_time = toc
        disp('done')
    end

    %M-step: determine optimal parameters given the sample set
    tic
    rom.M_step;
    M_step_time = toc
    rom.dispCurrentParams;
    iterations = rom.EM_iterations
    epochs = rom.epoch
    rom.linearFilterUpdate;
    
    plotTheta = feature('ShowFigureWindows');
    if plotTheta
        if ~exist('figureTheta')
            figureTheta =...
                figure('units','normalized','outerposition',[0 0 .5 1]);
        end
        rom.plotTheta(figureTheta);
    end
    
    if(~rom.conductivityTransformation.anisotropy)
        nFeatures = size(rom.designMatrix{1}, 2);
        Lambda_eff1_mode = conductivityBackTransform(...
            rom.designMatrix{1}(1:rom.coarseMesh.nEl, 1:nFeatures)...
            *rom.theta_c.theta(1:nFeatures), rom.conductivityTransformation)
    end
    
    %collect data and write it to disk periodically to save memory
    collectData;
    if(rom.epoch > rom.maxEpochs)
        break;
    end
end
clear tempArray;
runtime = toc


rom.predict;

predMetrics.meanSqDist = rom.meanSquaredDistance;
predMetrics.squaredDistance = rom.squaredDistance;
predMetrics.normError = rom.normError;
predMetrics.meanLogLikelihood = rom.meanLogLikelihood;
predMetrics.meanPerp = rom.meanPerplexity;
predMetrics.maha = rom.meanMahalanobisError;
predMetrics.meanSqDistField = rom.meanSquaredDistanceField;

%save predictions
save('./predictions.mat', 'predMetrics');






























