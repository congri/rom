%% Main script for 2d coarse-graining
%% Preamble
tic;    %measure runtime
clear all;
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

rng('shuffle')  %system time seed

%% Load training data
loadTrainingData;
%Get model and training parameters
params;

%Open parallel pool
parPoolInit(nTrain);
ppool = gcp;    %parallel pool properties
pend = 0;       %for sequential qi-updates
%prealloc
XMean = zeros(domainc.nEl, nTrain);
XSqMean = ones(domainc.nEl, nTrain);
XNormSqMean = ones(1, nTrain);

Tf = Tffile.Tf(:, nStart:(nStart + nTrain - 1));        %Finescale temperatures - load partially to save memory

%% Compute design matrices
Phi = DesignMatrix([domainf.nElX domainf.nElY], [domainc.nElX domainc.nElY], phi, Tffile, nStart:(nStart + nTrain - 1));
Phi = Phi.computeDesignMatrix(domainc.nEl, domainf.nEl, condTransOpts, mode);
%Normalize design matrices
%Phi = Phi.standardizeDesignMatrix;
Phi = Phi.rescaleDesignMatrix;
Phi.saveNormalization('rescaling'); %'rescaling' if rescaleDesignMatrix is used, 'standardization' if standardizeDesignMatrix is used
%Compute sum_i Phi^T(x_i)^Phi(x_i)
if strcmp(mode, 'useNeighbor')
    %use feature function information from nearest neighbors
    Phi = Phi.includeNearestNeighborFeatures([domainc.nElX domainc.nElY]);
elseif strcmp(mode, 'useLocalNeighbor')
    Phi = Phi.includeLocalNearestNeighborFeatures([domainc.nElX domainc.nElY]);
elseif strcmp(mode, 'useLocalDiagNeighbor')
    Phi = Phi.includeLocalDiagNeighborFeatures([domainc.nElX domainc.nElY]);
elseif strcmp(mode, 'useDiagNeighbor')
    %use feature function information from nearest and diagonal neighbors
    Phi = Phi.includeDiagNeighborFeatures([domainc.nElX domainc.nElY]);
elseif strcmp(mode, 'useLocal')
    %Use separate parameters for every macro-cell
    Phi = Phi.localTheta_c([domainc.nElX domainc.nElY]);
end
Phi = Phi.computeSumPhiTPhi;
if strcmp(mode, 'useLocal')
    Phi.sumPhiTPhi = sparse(Phi.sumPhiTPhi);
end

if theta_c.useNeuralNet
    finePerCoarse = [sqrt(size(Phi.xk{1}, 1)), sqrt(size(Phi.xk{1}, 1))];
    xkNN = zeros(finePerCoarse(1), finePerCoarse(2), 1, nTrain*domainc.nEl);
    k = 1;
    for i = 1:nTrain
        for j = 1:domainc.nEl
            xkNN(:, :, 1, k) =...
                reshape(Phi.xk{i}(:, j), finePerCoarse(1), finePerCoarse(2)); %for neural net
            k = k + 1;
        end
    end
end



%% EM optimization - main body
k = 1;          %EM iteration index
collectData;    %Write initial parametrizations to disk

for k = 2:(maxIterations + 1)

    %% Establish distribution to sample from
    for i = 1:nTrain
        Tf_i_minus_mu = Tf(:, i) - theta_cf.mu;
        if theta_c.useNeuralNet
            PhiMat = xkNN(:, :, 1, ((i - 1)*domainc.nEl + 1):(i*domainc.nEl));
        else
            PhiMat = Phi.designMatrices{i};
        end
        log_qi{i} = @(Xi) log_q_i(Xi, Tf_i_minus_mu, theta_cf, theta_c,...
            PhiMat, domainc, condTransOpts);
    end
    clear PhiMat;
    
    
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
        
        if (strcmp(update_qi, 'sequential') && k > 2)
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
        VIparams.optParams.stepWidth = stepWidthDropRate*VIparams.optParams.stepWidth;
        if(VIparams.optParams.stepWidth < stepWidthLowerBound)
            VIparams.optParams.stepWidth = stepWidthLowerBound;
        end
        parfor i = pstart:pend
            [optVarDist{i}, RMsteps{i}] = variationalInference(log_qi{i}, VIparams, initialParamsArray{i});
            initialParamsArray{i} = optVarDist{i}.params;
            if(strcmp(VIparams.family, 'diagonalGaussian'))
                VIdim = length(optVarDist{i}.params);
                XMean(:, i) = optVarDist{i}.params(1:(VIdim/2));
                XSqMean(:, i) = optVarDist{i}.params(1:(VIdim/2)).^2 + exp(-optVarDist{i}.params(((VIdim/2) + 1):end));
                XNormSqMean(i) = sum([optVarDist{i}.params(1:(VIdim/2)).^2 exp(-optVarDist{i}.params(((VIdim/2) + 1):end))]);
            else
                error('VI not implemented for this family of functions')
            end
        end
        save('./data/initialParamsArray.mat', 'initialParamsArray');
        %Set start values for next iteration
        for i = pstart:pend
            VIparams.initialParams{i} = optVarDist{i}.params;
        end
        disp('done')
        %Sample from VI distributions and solve coarse model
        for i = pstart:pend
            Tf_i_minus_mu = Tf(:, i) - theta_cf.mu;
            if(strcmp(VIparams.family, 'diagonalGaussian'))
                VIdim = length(optVarDist{1}.params);
                VImean = optVarDist{i}.params(1:(VIdim/2));
                VIsigma = exp(-.5*optVarDist{i}.params(((VIdim/2) + 1):end));
                %Samples of conductivity
                samples = conductivityBackTransform(mvnrnd(VImean, VIsigma, VIparams.inferenceSamples),...
                    condTransOpts);
                if condTransOpts.anisotropy
                    for j = 1:domainc.nEl
                        VIsamples{j} = mvnrnd(VImean((1 + (j - 1)*3):(j*3)),...
                            VIsigma((1 + (j - 1)*3):(j*3)), VIparams.inferenceSamples);
                    end
                else
                    % samples = logCond2Cond(mvnrnd(VImean, VIsigma, VIparams.inferenceSamples), 1e-10, 1e10);
                end
                
                for s = 1:VIparams.inferenceSamples
                    for j = 1:domainc.nEl
                        if condTransOpts.anisotropy
                            D(:, :, j) = conductivityBackTransform(VIsamples{j}(s, :)', condTransOpts);
                        else
                            D(:, :, j) =  samples(s, j)*eye(2);
                        end
                    end
                    FEMout = heat2d(domainc, D);
                    
                    Tc = FEMout.Tff';
                    Tc_samples(:, s, i) = Tc(:);
                end
%                 infRelErr = (std(Tc_samples, 0, 2)/sqrt(VIparams.inferenceSamples))./mean(Tc_samples, 2)
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
    lowerBoundS = eps;
    theta_cf.S = (1 - mix_S)*mean(p_cf_exponent, 2)...
        + mix_S*theta_cf.S + lowerBoundS*ones(domainf.nNodes, 1);
%     clear p_cf_exponent;
    theta_cf.Sinv = sparse(1:domainf.nNodes, 1:domainf.nNodes, 1./theta_cf.S);
    theta_cf.Sinv_vec = 1./theta_cf.S;
    theta_cf.WTSinv = theta_cf.W'*theta_cf.Sinv;        %Precomputation for efficiency

    %optimal theta_c and sigma
    Sigma_old = theta_c.Sigma;
    theta_old = theta_c.theta;
    %Adjust hyperprior to not get stuck at 0
    if(k - 1 <= size(theta_prior_hyperparamArray, 1))
        theta_prior_hyperparam = theta_prior_hyperparamArray(k - 1, :)
    end
    
    if(k > fixSigInit)
        sigma_prior_type = sigma_prior_type_hold;
    else
        sigma_prior_type = 'delta';
    end
    
    theta_c = optTheta_c(theta_c, nTrain, domainc.nEl, XSqMean,...
        Phi, XMean, theta_prior_type, theta_prior_hyperparam,...
        sigma_prior_type, sigma_prior_hyperparam);
    theta_c.Sigma = (1 - mix_sigma)*theta_c.Sigma + mix_sigma*Sigma_old;
    
    k
    if(~theta_c.useNeuralNet)
        theta_c.theta = (1 - mix_theta)*theta_c.theta + mix_theta*theta_old;
        disp('M-step done, current params:')
        [~, index] = sort(abs(theta_c.theta));
        if strcmp(mode, 'useNeighbor')
            feature = mod((index - 1), numel(Phi.featureFunctions)) + 1;
            neighborElement = floor((index - 1)/numel(Phi.featureFunctions)); %counted counterclockwise from right to lower neighbor
            curr_theta = [theta_c.theta(index) feature neighborElement]
        elseif strcmp(mode, 'useDiagNeighbor')
            feature = mod((index - 1), numel(Phi.featureFunctions)) + 1;
            neighborElement = floor((index - 1)/numel(Phi.featureFunctions)); %counted counterclockwise from right to lower right neighbor
            curr_theta = [theta_c.theta(index) feature neighborElement]
        elseif strcmp(mode, 'useLocal')
            feature = mod((index - 1), numel(Phi.featureFunctions)) + 1;
            Element = floor((index - 1)/numel(Phi.featureFunctions)) + 1;
            curr_theta = [theta_c.theta(index) feature Element]
        elseif(strcmp(mode, 'useLocalNeighbor') || strcmp(mode, 'useLocalDiagNeighbor'))
            disp('theta feature coarseElement neighbor')
            curr_theta = [theta_c.theta(index) Phi.neighborDictionary(index, 1)...
                Phi.neighborDictionary(index, 2) Phi.neighborDictionary(index, 3)]
        else
            curr_theta = [theta_c.theta(index) index]
        end
        
        if(linFiltSeq && k > initialIterations && mod(k, basisUpdateGap) == 0)
            %Construct E-matrix, see notes
            %        finePerCoarse = domainf.nEl/domainc.nEl;     %finescale pixels per coarse element - update this for non-square meshes!
            %        E = zeros(finePerCoarse);
            sigma2Inv_vec = (1./diag(theta_c.Sigma));
            XMeanMinusPhiThetac = zeros(domainc.nEl, nTrain);
            for i = 1:nTrain
                XMeanMinusPhiThetac(:, i) = XMean(:, i) - Phi.designMatrices{i}*theta_c.theta;
            end
            xk = Phi.xk;
            
            %        tic
            %        parfor i = 1:nTrain
            %            for j = 1:nTrain
            %                for m = 1:domainc.nEl
            %                    for n = 1:domainc.nEl
            %                        E = E + sigma2Inv_mat(m, n)*...
            %                            XMeanMinusPhiThetac{i}(m)*XMeanMinusPhiThetac{j}(n)*...
            %                            xk{i}(:, m)*xk{j}(:, n)';
            %                    end
            %                end
            %            end
            %        end
            %        assemble_t = toc
            
            xtemp = 0;
            for i = 1:nTrain     %serial seems to be more efficient here
                for m = 1:domainc.nEl
                    xtemp = xtemp + sigma2Inv_vec(m)*XMeanMinusPhiThetac(m, i)*xk{i}(:, m);
                end
            end
            
            pseudoinverse = false;
            if pseudoinverse
                Atemp = 0;
                for i = 1:nTrain     %serial seems to be more efficient here
                    for m = 1:domainc.nEl
                        Atemp = Atemp + sigma2Inv_vec(m)*(xk{i}(:, m)*xk{i}(:, m)');
                    end
                end
                xtemp = pinv(Atemp)*xtemp;
            end
            xtempNorm = norm(xtemp);
            E = xtempNorm^2
            xtemp = xtemp'/xtempNorm;
            
            %save xtemp
            filename = './data/w';
            save(filename, 'xtemp', '-ascii', '-append');
            %save E
            filename = './data/E';
            save(filename, 'E', '-ascii', '-append');
            
            %sigma
            filename = './data/sigma';
            sigma = full(diag(theta_c.Sigma))';
            save(filename, 'sigma', '-ascii', '-append');
            %        E = xtemp*xtemp';
            %
            %
            %        opts.issym = true;
            %        opts.isreal = true;
            %        [V, eigVals] = eigs(E, 4, 'lm', opts);
            %        V = real(V);
            %        eigVals = real(eigVals);
            %        figure(2)
            %        eigVals = diag(eigVals)
            %        [eigVals, indEigVals] = sort(eigVals, 'descend');
            %        V = V(:, indEigVals);
            %        plot(eigVals)
            %        figure(4)
            %        for i = 1:4
            %            subplot(2,2,i);
            %            imagesc(reshape(V(:, i), 64, 64))
            %            axis square
            %            grid off
            %            xticks({})
            %            yticks({})
            %            colorbar
            %        end
            figure(3)
            imagesc(reshape(xtemp, 64, 64))
            axis square
            grid off
            xticks({})
            yticks({})
            colorbar
            drawnow
            
            %        phi{end + 1} = @(lambda) sum(V(:, 1).*lambda);
            phi{end + 1} = @(lambda) sum(xtemp'.*log(lambda));
            nBasis = nBasis + 1;
            %% Compute design matrices
            %ATTENTION: this can and should be done more efficiently. Up to now,
            %all feature functions are recomputed
            Phi = DesignMatrix([domainf.nElX domainf.nElY], [domainc.nElX domainc.nElY], phi, Tffile, nStart:(nStart + nTrain - 1));
            Phi = Phi.computeDesignMatrix(domainc.nEl, domainf.nEl, condTransOpts);
            %Normalize design matrices
            %Phi = Phi.standardizeDesignMatrix;
%             Phi = Phi.rescaleDesignMatrix;
%             Phi.saveNormalization('rescaling'); %'rescaling' if rescaleDesignMatrix is used, 'standardization' if standardizeDesignMatrix is used
            %Compute sum_i Phi^T(x_i)^Phi(x_i)
            if strcmp(mode, 'useNeighbor')
                %use feature function information from nearest neighbors
                Phi = Phi.includeNearestNeighborFeatures([domainc.nElX domainc.nElY]);
            elseif strcmp(mode, 'useDiagNeighbor')
                %use feature function information from nearest and diagonal neighbors
                Phi = Phi.includeDiagNeighborFeatures([domainc.nElX domainc.nElY]);
            elseif strcmp(mode, 'useLocal')
                Phi = Phi.localTheta_c([domainc.nElX domainc.nElY]);
            end
            Phi = Phi.computeSumPhiTPhi;
            if strcmp(mode, 'useLocal')
                Phi.sumPhiTPhi = sparse(Phi.sumPhiTPhi);
            end
            %append theta-value
            if pseudoinverse
                theta_c.theta = [theta_c.theta; 1/xtempNorm];
            else
                theta_c.theta = [theta_c.theta; 0];
            end
            
        end
        
        plotTheta = true;
        if plotTheta
            if ~exist('thetaArray')
                thetaArray = theta_c.theta';
            else
                if(size(theta_c.theta, 1) > size(thetaArray, 2))
                    %New basis function included. Expand array
                    thetaArray = [thetaArray, zeros(size(thetaArray, 1), 1)];
                    thetaArray = [thetaArray; theta_c.theta'];
                else
                    thetaArray = [thetaArray; theta_c.theta'];
                end
            end
            if ~exist('sigmaArray')
                sigmaArray = diag(theta_c.Sigma)';
            else
                sigmaArray = [sigmaArray; diag(theta_c.Sigma)'];
            end
            figure(1)
            subplot(2,2,1)
            plot(thetaArray, 'linewidth', 1)
            axis tight;
            subplot(2,2,2)
            plot(theta_c.theta, 'linewidth', 1)
            %        imagesc(reshape(theta_c.theta, 64, 64))
            %        colorbar
            %        grid off;
            axis tight;
            subplot(2,2,3)
            semilogy(sqrt(sigmaArray), 'linewidth', 1)
            axis tight;
            %        ylim([0 20])
            subplot(2,2,4)
            imagesc(reshape(diag(sqrt(theta_c.Sigma)), domainc.nElX, domainc.nElY))
            title('\sigma_k')
            colorbar
            grid off;
            axis tight;
            drawnow
        end
    end
    
    curr_sigma = theta_c.Sigma
    mean_S = mean(theta_cf.S)
    if(~condTransOpts.anisotropy)
        if theta_c.useNeuralNet
            m = predict(theta_c.theta, xkNN(:, :, 1, 1:domainc.nEl));
            Lambda_eff1_mode = conductivityBackTransform(m, condTransOpts)
        else
            Lambda_eff1_mode = conductivityBackTransform(Phi.designMatrices{1}*theta_c.theta,...
                condTransOpts)
        end
    end
    
    %collect data and write it to disk periodically to save memory
    collectData;
end
%tidy up
clear i j k m Wa Wa_mean Tc_dyadic_mean log_qi p_cf_exponent curr_theta XMean XNormSqMean;
runtime = toc

































