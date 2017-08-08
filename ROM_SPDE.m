classdef ROM_SPDE
    %Class for ROM SPDE model
    
    properties  %public
        %% Finescale data specifications
        %Finescale system size
        nElFX = 256;
        nElFY = 256;
        %Finescale conductivities, binary material
        lowerConductivity = 1;
        upperConductivity = 10;
        %Conductivity field distribution type
        conductivityDistribution = 'correlated_binary';
        %Boundary condition functions; evaluate those on boundaries to get boundary conditions
        boundaryTemperature;
        boundaryHeatFlux;
        %Directory where finescale data is stored; specify basename here
        fineScaleDataPath = '~/matlab/data/fineData/';
        %matfile handle
        trainingDataMatfile;
        testDataMatfile;
        %Finescale Domain object
        fineScaleDomain;
        %Array holding fine scale data output; possibly large
        fineScaleDataOutput;
        %number of samples per generated matfile
        nSets = [1024 256];
        %Output data characteristics
        outputVariance;
        meanOutputVariance;
        E;  %Mapping from fine to coarse cell index
        neighborDictionary; %Gives neighbors of macrocells
                
        %% Model training parameters
        nStart = 1;             %first training data sample in file
        nTrain = 64;            %number of samples used for training
        mode = 'none';          %useNeighbor, useLocalNeighbor, useDiagNeighbor, useLocalDiagNeighbor, useLocal, global
                                %global: take whole microstructure as feature function input, not
                                %only local window (only recommended for pooling)
        inferenceMethod = 'variationalInference';        %E-step inference method. variationalInference or monteCarlo
        
        %% Sequential addition of linear filters
        linFilt
        
        useAutoEnc = true;     %Use autoencoder information? Do not forget to pre-train autoencoder!
        secondOrderTerms;
        mix_S = 0;              %To slow down convergence of S
        mix_theta = 0;
        mix_sigma = 0;
        fixSigmaInit = 0;       %Fix sigma in initial iterations
        
        %% Prior specifications
        sigmaPriorType = 'none';    %none, delta, expSigSq
        sigmaPriorHyperparam = 1;
        thetaPriorType = 'RVM';    %none, gaussian, RVM, hierarchical_laplace, hierarchical_gamma
        thetaPriorHyperparam = .1;
        
        %% Model parameters
        theta_c;
        theta_cf;
        featureFunctions;       %Cell array containing local feature function handles
        globalFeatureFunctions  %cell array with handles to global feature functions
        kernelHandles
        %transformation of finescale conductivity to real axis
        conductivityTransformation;
        latentDim;              %If autoencoder is used
        sumPhiTPhi;             %Design matrix precomputation
        
        %% Feature function rescaling parameters
        featureScaling = 'normalize'; %'standardize' for zero mean and unit variance of features, 'rescale' to have
                                        %all between 0 and 1, 'normalize' to have unit variance only
                                        %(and same mean as before)
        featureFunctionMean;
        featureFunctionSqMean;
        featureFunctionMin;
        featureFunctionMax;
        loadDesignMatrix = false;
        useKernels = true;              %Use linear combination of kernels in feature function space
        kernelBandwidth = 2;
        bandwidthSelection = 'silverman'  %'fixed' for fixed kernel bandwidth, 'silverman' for
                                            %tau = 1.06*sigma_i*n^(-1/5)
        kernelType = 'squaredExponential';
        
        %% Prediction parameters
        nSamples_p_c = 1000;
        testSamples;       %pick out specific test samples here
        trainingSamples;   %pick out specific training samples here
        
        %% Prediction outputs
        predMeanArray;
        predVarArray;
        meanPredMeanOutput;                %mean predicted mean of output field
        meanSquaredDistance;               %mean squared distance of predicted mean to true solution
        meanSquaredDistanceError;          %Monte Carlo error
        meanMahalanobisError;
        
        %% Finescale data- only load this to memory when needed!
        lambdak
        xk
        
        %% Computational quantities
        varExpect_p_cf_exp;     %Expected exponent of p_cf under q
        XMean;                  %Expected value of X under q
        XSqMean;                %<X^2>_q
        thetaArray;
        thetaHyperparamArray;
        sigmaArray;
        
        EM_iterations = 1;
        epoch = 0;      %how often every data point has been seen
        epoch_old = 0;  %To check if epoch has changed
        maxEpochs;      %Maximum number of epochs
    end
    
    
    properties(SetAccess = private)
        %% finescale data specifications
        conductivityLengthScaleDist = 'lognormal';      %delta for fixed length scale, lognormal for rand
        conductivityDistributionParams = {-1 [-3 .5] 1};     %for correlated_binary:
        %{volumeFraction, correlationLength, sigma_f2}
        %for log normal length scale, the
        %length scale parameters are log normal mu and
        %sigma
        %Coefficients giving boundary conditions, specify as string
        boundaryConditions = '[0 1000 0 0]';
        
        %% Coarse model specifications
        coarseScaleDomain;
        coarseGridVectorX = [1/4 1/4 1/4 1/4];
        coarseGridVectorY = [1/4 1/4 1/4 1/4];
        
        %Design matrices. Cell index gives data point, row index coarse cell, and column index
        %feature function
        designMatrix
        kernelMatrix
    end
    
    
    methods
        function obj = ROM_SPDE()
            %Constructor
            %Create data directory
            if ~exist('./data/', 'dir')
                mkdir('./data/');
            end
            %set up path
            obj = obj.genFineScaleDataPath;
            
            %Set handles to boundary condition functions
            obj = obj.genBoundaryConditionFunctions;
            
            %Set up coarseScaleDomain; must be done after boundary conditions are set up
            obj = obj.genCoarseDomain;
            
            %prealloc
            obj.XMean = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
            obj.XSqMean = ones(obj.coarseScaleDomain.nEl, obj.nTrain);
            
            %Set up default value for test samples
            obj.testSamples = 1:obj.nSets(2);
            obj.trainingSamples = obj.nStart:(obj.nStart + obj.nTrain - 1);
            
            %Set conductivity transformation
            obj.conductivityTransformation.anisotropy = false;
            obj.conductivityTransformation.type = 'logit';
            if strcmp(obj.conductivityTransformation.type, 'log')
                obj.conductivityTransformation.limits = [1e-8 1e8];
            elseif strcmp(obj.conductivityTransformation.type, 'logit')
                obj.conductivityTransformation.limits =...
                    [(1 - 1e-4)*obj.lowerConductivity (1 + 1e-4)*obj.upperConductivity];
            else
                obj.conductivityTransformation.limits = [1e-8 1e8];
            end
            conductivityTransformation = obj.conductivityTransformation;
            save('./data/conductivityTransformation', 'conductivityTransformation');
            
            %Set up feature function handles
            obj = obj.setFeatureFunctions;
        end
        
        function obj = genFineScaleData(obj, boundaryConditions, condDistParams)
            %Function to generate and save finescale data
            
            disp('Generate fine scale data...')
            
            if(nargin > 1)
                obj = obj.setBoundaryConditions(boundaryConditions);
            end
            
            if(nargin > 2)
                obj = obj.setConductivityDistributionParams(condDistParams);
            end
            
            %for boundary condition functions
            if(isempty(obj.boundaryTemperature) || isempty(obj.boundaryHeatFlux))
                obj = obj.genBoundaryConditionFunctions;
            end
            
            %% Generate finescale domain
            tic
            disp('Generate finescale domain...')
            addpath('./heatFEM')    %to find Domain class
            obj.fineScaleDomain = Domain(obj.nElFX, obj.nElFY);
            obj.fineScaleDomain = setBoundaries(obj.fineScaleDomain, [2:(2*obj.nElFX + 2*obj.nElFY)],...
                obj.boundaryTemperature, obj.boundaryHeatFlux);       %Only fix lower left corner as essential node
            disp('done')
            domain_generating_time = toc
            
            if ~exist(obj.fineScaleDataPath, 'dir')
                mkdir(obj.fineScaleDataPath);
            end
            
            %Generate finescale conductivity samples and solve FEM
            for i = 1:numel(obj.nSets)
                filename = strcat(obj.fineScaleDataPath, 'set', num2str(i), '-samples=', num2str(obj.nSets(i)));
                obj.solveFEM(i, filename);
            end
            
            %save params
            fineScaleDomain = obj.fineScaleDomain;
            save(strcat(obj.fineScaleDataPath, 'fineScaleDomain.mat'), 'fineScaleDomain');
            
            disp('done')
        end
        
        function obj = genBoundaryConditionFunctions(obj)
            %Set up boundary condition functions
            if isempty(obj.boundaryConditions)
                error('No string specified for boundary conditions')
            end
            bc = str2num(obj.boundaryConditions);
            obj.boundaryTemperature = @(x) bc(1) + bc(2)*x(1) + bc(3)*x(2) + bc(4)*x(1)*x(2);
            obj.boundaryHeatFlux{1} = @(x) -(bc(3) + bc(4)*x);      %lower bound
            obj.boundaryHeatFlux{2} = @(y) (bc(2) + bc(4)*y);       %right bound
            obj.boundaryHeatFlux{3} = @(x) (bc(3) + bc(4)*x);       %upper bound
            obj.boundaryHeatFlux{4} = @(y) -(bc(2) + bc(4)*y);      %left bound
        end
        
        function cond = generateConductivityField(obj, nSet)
            %nSet is the number of the data set
            %nSet is the set (file number) index
            
            % Draw conductivity/ log conductivity
            disp('Generating finescale conductivity field...')
            tic
            if strcmp(obj.conductivityDistribution, 'uniform')
                %conductivity uniformly distributed between lo and up
                cond{1} = zeros(obj.fineScaleDomain.nEl, 1);
                cond = repmat(cond, 1, obj.nSets(nSet));
                for i = 1:obj.nSets(nSet)
                    cond{i} = (obj.upperConductivity - obj.lowerConductivity)*...
                        rand(obj.fineScaleDomain.nEl, 1) + obj.lowerConductivity;
                end
            elseif strcmp(obj.conductivityDistribution, 'gaussian')
                %log conductivity gaussian distributed
                x = normrnd(obj.conductivityDistributionParams{1}, obj.conductivityDistributionParams{2},...
                    obj.fineScaleDomain.nEl, obj.nSets(nSet));
                cond{1} = zeros(obj.fineScaleDomain.nEl, 1);
                cond = repmat(cond, 1, obj.nSets(nSet));
                for i = 1:obj.nSets(nSet)
                    cond{i} = exp(x(:, i));
                end
            elseif strcmp(obj.conductivityDistribution, 'binary')
                %binary distribution of conductivity (Bernoulli)
                cond{1} = zeros(obj.fineScaleDomain.nEl, 1);
                cond = repmat(cond, 1, obj.nSets(nSet));
                for i = 1:obj.nSets(nSet)
                    r = rand(obj.fineScaleDomain.nEl, 1);
                    cond{i} = obj.lowerConductivity*ones(obj.fineScaleDomain.nEl, 1);
                    cond{i}(r < obj.conductivityDistributionParams{1}) = obj.upperConductivity;
                end
            elseif strcmp(obj.conductivityDistribution, 'correlated_binary')
                %ATTENTION: so far, only isotropic distributions (length scales) possible
                %Compute coordinates of element centers
                x = .5*(obj.fineScaleDomain.cum_lElX(1:(end - 1)) + obj.fineScaleDomain.cum_lElX(2:end));
                y = .5*(obj.fineScaleDomain.cum_lElY(1:(end - 1)) + obj.fineScaleDomain.cum_lElY(2:end));
                [X, Y] = meshgrid(x, y);
                %directly clear potentially large arrays
                clear y;
                x = [X(:) Y(:)]';
                clear X Y;
                
                addpath('./computation')        %to find parPoolInit
                parPoolInit(obj.nSets(nSet));
                %Store conductivity fields in cell array to avoid broadcasting the whole data
                cond{1} = zeros(obj.fineScaleDomain.nEl, 1);
                cond = repmat(cond, 1, obj.nSets(nSet));
                
                addpath('./genConductivity')        %to find genBochnerSamples
                nBochnerBasis = 1e3;    %Number of cosine basis functions
                for i = 1:(obj.nSets(nSet))
                    if strcmp(obj.conductivityLengthScaleDist, 'delta')
                        %one fixed length scale for all samples
                        l = obj.conductivityDistributionParams{2}(1);
                    elseif strcmp(obj.conductivityLengthScaleDist, 'lognormal')
                        %First and second parameters are mu and sigma of lognormal dist
                        l = lognrnd(obj.conductivityDistributionParams{2}(1),...
                            obj.conductivityDistributionParams{2}(2));
                    else
                        error('Unknown length scale distribution')
                    end
                    p{i} = genBochnerSamples(l, obj.conductivityDistributionParams{3},...
                        nBochnerBasis);
                end
                nEl = obj.fineScaleDomain.nEl;
                upCond = obj.upperConductivity;
                loCond = obj.lowerConductivity;
                %set volume fraction parameter < 0 to have uniformly random volume fraction
                if(obj.conductivityDistributionParams{1} >= 0)
                    cutoff = norminv(1 - obj.conductivityDistributionParams{1},...
                        0, obj.conductivityDistributionParams{3});
                else
                    cutoff = zeros(obj.nSets(nSet), 1);
                    for i = 1:(obj.nSets(nSet))
                        phiRand = rand;
                        cutoff(i) = norminv(1 - phiRand, 0, obj.conductivityDistributionParams{3});
                    end
                end
                volfrac = obj.conductivityDistributionParams{1};
                parfor i = 1:(obj.nSets(nSet))
                    %use for-loop instead of vectorization to save memory
                    for j = 1:nEl
                        ps = p{i}(x(:, j));
                        if(volfrac >= 0)
                            cond{i}(j) = upCond*(ps > cutoff) + loCond*(ps <= cutoff);
                        else
                            cond{i}(j) = upCond*(ps > cutoff(i)) + loCond*(ps <= cutoff(i));
                        end
                    end
                end
            else
                error('unknown FOM conductivity distribution');
            end
            disp('done')
            conductivity_generation_time = toc
        end
        
        function obj = solveFEM(obj, nSet, savepath)
            
            cond = obj.generateConductivityField(nSet);
            %Solve finite element model
            disp('Solving finescale problem...')
            tic
            Tf = zeros(obj.fineScaleDomain.nNodes, obj.nSets(nSet));
            D{1} = zeros(2, 2, obj.fineScaleDomain.nEl);
            D = repmat(D, obj.nSets(nSet), 1);
            domain = obj.fineScaleDomain;   %To avoid broadcasting overhead
            parPoolInit(obj.nSets(nSet));
            parfor i = 1:obj.nSets(nSet)
                %Conductivity matrix D, only consider isotropic materials here
                for j = 1:domain.nEl
                    D{i}(:, :, j) =  cond{i}(j)*eye(2);
                end
                FEMout = heat2d(domain, D{i});
                %Store fine temperatures as a vector Tf. Use reshape(Tf(:, i), domain.nElX + 1, domain.nElY + 1)
                %and then transpose result to reconvert it to original temperature field
                Ttemp = FEMout.Tff';
                Tf(:, i) = Ttemp(:);
            end
            disp('FEM systems solved')
            tot_FEM_time = toc
            
            if(nargin > 2)
                disp('saving finescale data...')
                cond = cell2mat(cond);
                save(strcat(savepath, ''), 'cond', 'Tf', '-v7.3')    %partial loading only for -v7.3
                disp('done')
            end
        end
        
        function obj = genFineScaleDataPath(obj)
            volFrac = obj.conductivityDistributionParams{1};
            sigma_f2 = obj.conductivityDistributionParams{3};
            obj.fineScaleDataPath = strcat(obj.fineScaleDataPath,...
                'systemSize=', num2str(obj.nElFX), 'x', num2str(obj.nElFY), '/');
            %Type of conductivity distribution
            if strcmp(obj.conductivityDistribution, 'correlated_binary')
                if strcmp(obj.conductivityLengthScaleDist, 'delta')
                    corrLength = obj.conductivityDistributionParams{2}(1);
                    obj.fineScaleDataPath = strcat(obj.fineScaleDataPath,...
                        obj.conductivityDistribution, '/', 'IsoSEcov/', 'l=',...
                        num2str(corrLength), '_sigmafSq=', num2str(sigma_f2),...
                        '/volumeFraction=', num2str(volFrac), '/', 'locond=',...
                        num2str(obj.lowerConductivity), '_upcond=', num2str(obj.upperConductivity),...
                        '/', 'BCcoeffs=', obj.boundaryConditions, '/');
                elseif strcmp(obj.conductivityLengthScaleDist, 'lognormal')
                    corrLength1 = obj.conductivityDistributionParams{2}(1);
                    corrLength2 = obj.conductivityDistributionParams{2}(2);
                    obj.fineScaleDataPath = strcat(obj.fineScaleDataPath,...
                        obj.conductivityDistribution, '/', 'IsoSEcov/', 'l=lognormal_mu=',...
                        num2str(corrLength1), 'sigma=', num2str(corrLength2),...
                        '_sigmafSq=', num2str(sigma_f2), '/volumeFraction=',...
                        num2str(volFrac), '/', 'locond=', num2str(obj.lowerConductivity),...
                        '_upcond=', num2str(obj.upperConductivity),...
                        '/', 'BCcoeffs=', obj.boundaryConditions, '/');
                else
                    error('Unknown length scale distribution')
                end
            elseif strcmp(cond_distribution, 'binary')
                obj.fineScaleDataPath = strcat(obj.fineScaleDataPath,...
                    obj.conductivityDistribution, '/volumeFraction=',...
                    num2str(volFrac), '/', 'locond=', num2str(obj.lowerConductivity),...
                    '_upcond=', num2str(obj.upperConductivity), '/', 'BCcoeffs=', obj.boundaryConditions, '/');
            else
                error('Unknown conductivity distribution')
            end
            %Name of training data file
            trainFileName = strcat('set1-samples=', num2str(obj.nSets(1)), '.mat');
            obj.trainingDataMatfile = matfile(strcat(obj.fineScaleDataPath, trainFileName));
            testFileName = strcat('set2-samples=', num2str(obj.nSets(2)), '.mat');
            obj.testDataMatfile = matfile(strcat(obj.fineScaleDataPath, testFileName));
        end
        
        function obj = loadTrainingData(obj)
            %load data params; warning for variable FD can be ignored
            try
                load(strcat(obj.fineScaleDataPath, 'fineScaleDomain.mat'));
                obj.fineScaleDomain = fineScaleDomain;
            catch
                temp = load(strcat(obj.fineScaleDataPath, 'romObj.mat'));
                obj.fineScaleDomain = temp.obj.fineScaleDomain;
            end
            %for finescale domain class
            addpath('./heatFEM')
            %for boundary condition functions
            if(isempty(obj.boundaryTemperature) || isempty(obj.boundaryHeatFlux))
                obj = obj.genBoundaryConditionFunctions;
            end
            
            %there is no cum_lEl (cumulated finite element length) in old data files
            if(~numel(obj.fineScaleDomain.cum_lElX) || ~numel(obj.fineScaleDomain.cum_lElX))
                obj.fineScaleDomain.cum_lElX = linspace(0, 1, obj.fineScaleDomain.nElX + 1);
                obj.fineScaleDomain.cum_lElY = linspace(0, 1, obj.fineScaleDomain.nElY + 1);
            end
            
            %load finescale temperatures partially
            obj.fineScaleDataOutput = obj.trainingDataMatfile.Tf(:, obj.nStart:(obj.nStart + obj.nTrain - 1));
        end
        
        function obj = genCoarseDomain(obj)
            %Generate coarse domain object
            nX = length(obj.coarseGridVectorX);
            nY = length(obj.coarseGridVectorY);
            addpath('./heatFEM')        %to find Domain class
            obj.coarseScaleDomain = Domain(nX, nY, obj.coarseGridVectorX, obj.coarseGridVectorY);
            %ATTENTION: natural nodes have to be set manually
            %and consistently in coarse and fine scale domain!!
            obj.coarseScaleDomain = setBoundaries(obj.coarseScaleDomain, [2:(2*nX + 2*nY)],...
                obj.boundaryTemperature, obj.boundaryHeatFlux);
            
            %Legacy, for predictions
            if ~exist('./data/', 'dir')
                mkdir('./data/');
            end
            filename = './data/coarseScaleDomain.mat';
            coarseScaleDomain = obj.coarseScaleDomain;
            save(filename, 'coarseScaleDomain');
        end
        
        function obj = estimateDataVariance(obj)
            
            Tftemp = obj.trainingDataMatfile.Tf(:, 1);
            Tf_true_mean = zeros(size(Tftemp));
            Tf_true_sq_mean = zeros(size(Tftemp));
            nSamples = obj.nSets(1);
            window = nSamples;
            nWindows = ceil(nSamples/window);
            tic
            for i = 1:nWindows
                initial = 1 + (i - 1)*window;
                final = i*window;
                if final > nSamples
                    final = nSamples;
                end
                Tftemp = obj.trainingDataMatfile.Tf(:, initial:final);
                Tf_mean_temp = mean(Tftemp, 2);
                Tf_sq_mean_temp = mean(Tftemp.^2, 2);
                clear Tftemp;
                Tf_true_mean = ((i - 1)/i)*Tf_true_mean + (1/i)*Tf_mean_temp;
                Tf_true_sq_mean = ((i - 1)/i)*Tf_true_sq_mean + (1/i)*Tf_sq_mean_temp;
            end
            
            Tf_true_var = Tf_true_sq_mean - Tf_true_mean.^2;
            obj.outputVariance = Tf_true_var;
            obj.meanOutputVariance = mean(Tf_true_var);
            toc
            
            sv = false;
            if sv
                %     savedir = '~/matlab/data/trueMC/';
                savedir = './';
                if ~exist(savedir, 'dir')
                    mkdir(savedir);
                end
                save(strcat(savedir, '', '_nSamples=', num2str(nSamples), '.mat'), 'Tf_true_mean', 'Tf_true_var')
            end
        end
        
        function [Xopt, LambdaOpt, s2] = detOpt_p_cf(obj, nStart, nTrain)
            %Deterministic optimization of log(p_cf) to check capabilities of model
            
            %don't change these!
            theta_cfOptim.S = 1;
            theta_cfOptim.sumLogS = 0;
            theta_cfOptim.Sinv = 1;
            theta_cfOptim.Sinv_vec = ones(obj.fineScaleDomain.nNodes, 1);
            theta_cfOptim.W = obj.theta_cf.W;
            theta_cfOptim.WTSinv = obj.theta_cf.WTSinv;
            
            options = optimoptions(@fminunc,'Display','iter', 'Algorithm', 'trust-region',...
                'SpecifyObjectiveGradient', true);
            Xinit = 0*ones(obj.coarseScaleDomain.nEl, 1);
            Xopt = zeros(obj.coarseScaleDomain.nEl, nTrain);
            LambdaOpt = Xopt;
            s2 = zeros(1, nTrain);
            j = 1;
            addpath('./tests/detOptP_cf')
            for i = nStart:(nStart + nTrain -1)
                Tf = obj.trainingDataMatfile.Tf(:, i);
                objFun = @(X) objective(X, Tf, obj.coarseScaleDomain, obj.conductivityTransformation, theta_cfOptim);
                [XoptTemp, fvalTemp] = fminunc(objFun, Xinit, options);
                LambdaOptTemp = conductivityBackTransform(XoptTemp, obj.conductivityTransformation);
                Xopt(:, j) = XoptTemp;
                LambdaOpt(:, j) = LambdaOptTemp;
                
                %s2 is the squared distance of truth to optimal coarse averaged over all nodes
                s2(j) = fvalTemp/obj.fineScaleDomain.nNodes
                j = j + 1;
            end
        end
        
        function obj = loadTrainedParams(obj)
            %Load trained model parameters from disk to workspace
            
            %Load trained params from disk
            disp('Loading optimal parameters from disk...')
            obj.theta_c.theta = dlmread('./data/theta');
            obj.theta_c.theta = obj.theta_c.theta(end, :)';
            obj.theta_c.Sigma = dlmread('./data/sigma');
            obj.theta_c.Sigma = diag(obj.theta_c.Sigma(end, :));
            obj.theta_cf.S = dlmread('./data/S')';
            W = dlmread('./data/Wmat');
            W = reshape(W, length(W)/3, 3)';
            obj.theta_cf.W = sparse(W(1, :), W(2, :), W(3, :));
            obj.theta_cf.mu = dlmread('./data/mu')';
            disp('done')
            
            disp('Loading data normalization data...')
            try
                obj.featureFunctionMean = dlmread('./data/featureFunctionMean');
                obj.featureFunctionSqMean = dlmread('./data/featureFunctionSqMean');
            catch
                warning('featureFunctionMean, featureFunctionSqMean not found, setting it to 0.')
                obj.featureFunctionMean = 0;
                obj.featureFunctionSqMean = 0;
            end
            
            try
                obj.featureFunctionMin = dlmread('./data/featureFunctionMin');
                obj.featureFunctionMax = dlmread('./data/featureFunctionMax');
            catch
                warning('featureFunctionMin, featureFunctionMax not found, setting it to 0.')
                obj.featureFunctionMin = 0;
                obj.featureFunctionMax = 0;
            end
            disp('done')
            
            if(isempty(obj.coarseScaleDomain) || isempty(obj.fineScaleDomain))
                disp('Loading fine and coarse domain objects...')
                addpath('./heatFEM')        %to find Domain class
                try
                    load(strcat(obj.fineScaleDataPath, 'fineScaleDomain.mat'));
                    obj.fineScaleDomain = fineScaleDomain;
                catch
                    temp = load(strcat(obj.fineScaleDataPath, 'romObj.mat'));
                    obj.fineScaleDomain = temp.obj.fineScaleDomain;
                end
                
                if exist(strcat('./data/coarseScaleDomain.mat'), 'file')
                    load(strcat('./data/coarseScaleDomain.mat'));
                    obj.coarseScaleDomain = coarseScaleDomain;
                else
                    warning(strcat('No coarse domain file found.',...
                        'Take boundary conditions from finescale data and regenerate.',...
                        'Please make sure everything is correct!'))
                    obj = obj.genCoarseDomain;
                end
                disp('done')
            end
            
            %Generate same basis functions as in training
            disp('Setting up function handles to p_c basis functions...')
            addpath('./params')
            obj = obj.setFeatureFunctions;
            nFeatures = size(obj.featureFunctions, 2);
            nGlobalFeatures = size(obj.globalFeatureFunctions, 2);
            if(obj.linFilt.totalUpdates > 0)
                if exist('./data/w.mat', 'file')
                    load('./data/w.mat');   %to load w_all
                    for i = 1:size(w_all, 2)
                        for m = 1:obj.coarseScaleDomain.nEl
                            obj.featureFunctions{m, nFeatures + 1} = @(lambda) sum(w_all{m, i}'.*...
                                conductivityTransform(lambda(:), obj.conductivityTransformation));
                        end
                        nFeatures = nFeatures + 1;
                    end
                end
                
                if exist('./data/wGlobal.mat', 'file')
                    load('./data/wGlobal.mat');   %to load w_allGlobal, i.e. global linear filters
                    for i = 1:size(w_allGlobal, 2)
                        for m = 1:obj.coarseScaleDomain.nEl
                            obj.globalFeatureFunctions{m, nGlobalFeatures + 1} = @(lambda) sum(w_allGlobal{m, i}'.*...
                                conductivityTransform(lambda(:), obj.conductivityTransformation));
                        end
                        nGlobalFeatures = nGlobalFeatures + 1;
                    end
                end
            end
            disp('done')
        end
        
        function obj = M_step(obj)
            disp('M-step: find optimal params...')
            %Optimal S (decelerated convergence)
            lowerBoundS = eps;
            obj.theta_cf.S = (1 - obj.mix_S)*mean(obj.varExpect_p_cf_exp, 2)...
                + obj.mix_S*obj.theta_cf.S + lowerBoundS*ones(obj.fineScaleDomain.nNodes, 1);
            obj.theta_cf.Sinv = sparse(1:obj.fineScaleDomain.nNodes,...
                1:obj.fineScaleDomain.nNodes, 1./obj.theta_cf.S);
            obj.theta_cf.Sinv_vec = 1./obj.theta_cf.S;
            obj.theta_cf.WTSinv = obj.theta_cf.W'*obj.theta_cf.Sinv;        %Precomputation for efficiency
            
            %optimal theta_c and sigma
            Sigma_old = obj.theta_c.Sigma;
            theta_old = obj.theta_c.theta;
            
            obj = obj.updateTheta_c;
%             obj.theta_c.thetaPriorType = obj.thetaPriorType;
%             obj.theta_c.priorHyperparam = obj.thetaPriorHyperparam;
%             
%             obj.theta_c = optTheta_c(obj.theta_c, obj.nTrain, obj.coarseScaleDomain.nEl, obj.XSqMean,...
%                 obj.designMatrix, obj.XMean, obj.sigmaPriorType, obj.sigmaPriorHyperparam)
            obj.theta_c.Sigma = (1 - obj.mix_sigma)*obj.theta_c.Sigma + obj.mix_sigma*Sigma_old;
            obj.theta_c.theta = (1 - obj.mix_theta)*obj.theta_c.theta + obj.mix_theta*theta_old;
            
            disp('M-step done')
        end
        
        function obj = updateTheta_c(obj)
            %Find optimal theta_c and sigma
            dim_theta = numel(obj.theta_c.theta);
            
            %% Solve self-consistently: compute optimal sigma2, then theta, then sigma2 again and so on
            %Start from previous best estimate
            theta = obj.theta_c.theta;
            I = speye(dim_theta);
            Sigma = obj.theta_c.Sigma;
            
            %sum_i Phi_i^T Sigma^-1 <X^i>_qi
            sumPhiTSigmaInvXmean = 0;
            SigmaInv = obj.theta_c.SigmaInv;
            SigmaInvXMean = SigmaInv*obj.XMean;
            sumPhiTSigmaInvPhi = 0;
            PhiThetaMat = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
            
            for n = 1:obj.nTrain
                sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + obj.designMatrix{n}'*SigmaInvXMean(:, n);
                sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + obj.designMatrix{n}'*SigmaInv*obj.designMatrix{n};
                PhiThetaMat(:, n) = obj.designMatrix{n}*obj.theta_c.theta;
            end
            
            if(strcmp(obj.thetaPriorType, 'gaussian') || strcmp(obj.thetaPriorType, 'RVM'))
                %Find prior hyperparameter by max marginal likelihood
                converged = false;
                iter = 0;
                if strcmp(obj.thetaPriorType, 'gaussian')
                    obj.thetaPriorHyperparam = 1;
                elseif strcmp(obj.thetaPriorType, 'RVM')
                    obj.thetaPriorHyperparam = ones(dim_theta, 1);
                end
                while(~converged)
                    stabilityParam = 1e-10;    %for stability in matrix inversion
                    if strcmp(obj.thetaPriorType, 'gaussian')
                        SigmaTilde = inv(sumPhiTSigmaInvPhi + (obj.thetaPriorHyperparam + stabilityParam)*I);
                        muTilde = SigmaTilde*sumPhiTSigmaInvXmean;
                        theta_prior_hyperparam_old = obj.thetaPriorHyperparam;
                        obj.thetaPriorHyperparam = dim_theta/(muTilde'*muTilde + trace(SigmaTilde));
                    elseif strcmp(obj.thetaPriorType, 'RVM')
                        SigmaTilde = inv(sumPhiTSigmaInvPhi + diag(obj.thetaPriorHyperparam + stabilityParam));
                        muTilde = SigmaTilde*sumPhiTSigmaInvXmean;
                        theta_prior_hyperparam_old = obj.thetaPriorHyperparam;
                        obj.thetaPriorHyperparam = 1./(muTilde.^2 + diag(SigmaTilde));
                    end
                    if(norm(obj.thetaPriorHyperparam - theta_prior_hyperparam_old)/norm(obj.thetaPriorHyperparam)...
                            < 1e-5 || iter > 1000)
                        converged = true;
                    elseif(any(~isfinite(obj.thetaPriorHyperparam)) || any(obj.thetaPriorHyperparam <= 0))
                        converged = true;
                        obj.thetaPriorHyperparam = ones(dim_theta, 1);
                        warning('Gaussian hyperparameter precision is negative or not a number. Setting it to 1.')
                    end
                    iter = iter + 1;
                end
            end
            
            iter = 0;
            converged = false;
            while(~converged)
                theta_old = theta;  %to check for iterative convergence
                
                %Matrix M is pos. def., invertible even if badly conditioned
                %warning('off', 'MATLAB:nearlySingularMatrix');
                if strcmp(obj.thetaPriorType, 'hierarchical_laplace')
                    offset = 1e-30;
                    U = diag(sqrt((abs(theta) + offset)/obj.thetaPriorHyperparam(1)));
                elseif strcmp(obj.thetaPriorType, 'hierarchical_gamma')
                    U = diag(sqrt((.5*abs(theta).^2 + obj.thetaPriorHyperparam(2))./...
                        (obj.thetaPriorHyperparam(1) + .5)));
                elseif strcmp(obj.thetaPriorType, 'gaussian')
                    sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + (obj.thetaPriorHyperparam(1) + stabilityParam)*I;
                elseif strcmp(obj.thetaPriorType, 'RVM')
                    sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + diag(obj.thetaPriorHyperparam + stabilityParam);
                elseif strcmp(obj.thetaPriorType, 'none')
                else
                    error('Unknown prior on theta_c')
                end
                
                if (strcmp(obj.thetaPriorType, 'gaussian') || strcmp(obj.thetaPriorType, 'RVM') ||...
                        strcmp(obj.thetaPriorType, 'none'))
                    theta_temp = sumPhiTSigmaInvPhi\sumPhiTSigmaInvXmean;
                else
                    theta_temp = U*((U*sumPhiTSigmaInvPhi*U + I)\U)*sumPhiTSigmaInvXmean;
                end
                [~, msgid] = lastwarn;     %to catch nearly singular matrix
                
                if(strcmp(msgid, 'MATLAB:singularMatrix') || strcmp(msgid, 'MATLAB:nearlySingularMatrix')...
                        || strcmp(msgid, 'MATLAB:illConditionedMatrix') || norm(theta_temp)/length(theta) > 1e8)
                    warning('theta_c is assuming unusually large values. Only go small step.')
                    theta = .5*(theta + .1*(norm(theta)/norm(theta_temp))*theta_temp)
                    if any(~isfinite(theta))
                        %restart from 0
                        warning('Some components of theta are not finite. Restarting from theta = 0...')
                        theta = 0*theta;
                    end
                else
                    theta = theta_temp;
                end
                
                PhiThetaMat = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
                for n = 1:obj.nTrain
                    PhiThetaMat(:, n) = obj.designMatrix{n}*theta;
                end
                if(obj.EM_iterations < obj.fixSigmaInit)
                    sigma_prior_type = 'delta';
                else
                    sigma_prior_type = obj.sigmaPriorType;
                end
                if strcmp(sigma_prior_type, 'none')
                    Sigma = sparse(1:obj.coarseScaleDomain.nEl, 1:obj.coarseScaleDomain.nEl,...
                        mean(obj.XSqMean - 2*(PhiThetaMat.*obj.XMean) +...
                        PhiThetaMat.^2, 2));
                    Sigma(Sigma < 0) = eps; %for numerical stability
                    
                    %sum_i Phi_i^T Sigma^-1 <X^i>_qi
                    sumPhiTSigmaInvXmean = 0;
                    %Only valid for diagonal Sigma
                    s = diag(Sigma);
                    SigmaInv = sparse(diag(1./s));
                    SigmaInvXMean = SigmaInv*obj.XMean;
                    sumPhiTSigmaInvPhi = 0;
                    
                    for n = 1:obj.nTrain
                        sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + obj.designMatrix{n}'*SigmaInvXMean(:, n);
                        sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + obj.designMatrix{n}'*SigmaInv*obj.designMatrix{n};
                    end
                elseif strcmp(sigma_prior_type, 'expSigSq')
                    %Vector of variances
                    sigmaSqVec = .5*(1./obj.sigmaPriorHyperparam).*...
                        (-.5*obj.nTrain + sqrt(2*obj.nTrain*obj.sigmaPriorHyperparam.*...
                        mean(obj.XSqMean - 2*(PhiThetaMat.*obj.XMean) + PhiThetaMat.^2, 2) + .25*obj.nTrain^2));
                    
                    Sigma = sparse(1:obj.coarseScaleDomain.nEl, 1:obj.coarseScaleDomain.nEl, sigmaSqVec);
                    Sigma(Sigma < 0) = eps; %for numerical stability
                    sumPhiTSigmaInvXmean = 0;
                    %Only valid for diagonal Sigma
                    s = diag(Sigma);
                    SigmaInv = sparse(diag(1./s));
                    SigmaInvXMean = SigmaInv*obj.XMean;
                    sumPhiTSigmaInvPhi = 0;
                    
                    for n = 1:obj.nTrain
                        sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + obj.designMatrix{n}'*SigmaInvXMean(:, n);
                        sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + obj.designMatrix{n}'*SigmaInv*obj.designMatrix{n};
                    end
                elseif strcmp(sigma_prior_type, 'delta')
                    %Don't change sigma
                    %sum_i Phi_i^T Sigma^-1 <X^i>_qi
                    sumPhiTSigmaInvXmean = 0;
                    %Only valid for diagonal Sigma
                    s = diag(Sigma);
                    SigmaInv = sparse(diag(1./s));
                    SigmaInvXMean = SigmaInv*obj.XMean;
                    sumPhiTSigmaInvPhi = 0;
                    
                    for n = 1:obj.nTrain
                        sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + obj.designMatrix{n}'*SigmaInvXMean(:, n);
                        sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + obj.designMatrix{n}'*SigmaInv*obj.designMatrix{n};
                    end
                end
                
                iter = iter + 1;
                thetaDiffRel = norm(theta_old - theta)/(norm(theta)*numel(theta));
                if((iter > 5 && thetaDiffRel < 1e-8) || iter > 200)
                    converged = true;
                end
            end
            
            obj.theta_c.theta = theta;
            % theta_c.sigma = sqrt(sigma2);
            obj.theta_c.Sigma = Sigma;
            obj.theta_c.SigmaInv = SigmaInv;
            obj.thetaPriorHyperparam = obj.thetaPriorHyperparam;
            noPriorSigma = mean(obj.XSqMean - 2*(PhiThetaMat.*obj.XMean) + PhiThetaMat.^2, 2);
            save('./data/noPriorSigma.mat', 'noPriorSigma');
            
        end
        
        function dispCurrentParams(obj)
            disp('Current params:')
            [~, index] = sort(abs(obj.theta_c.theta));
            if strcmp(obj.mode, 'useNeighbor')
                feature = mod((index - 1), numel(obj.featureFunctions)) + 1;
                %counted counterclockwise from right to lower neighbor
                neighborElement = floor((index - 1)/numel(obj.featureFunctions));
                curr_theta = [obj.theta_c.theta(index) feature neighborElement]
            elseif strcmp(obj.mode, 'useDiagNeighbor')
                feature = mod((index - 1), numel(obj.featureFunctions)) + 1;
                %counted counterclockwise from right to lower right neighbor
                neighborElement = floor((index - 1)/numel(obj.featureFunctions));
                curr_theta = [obj.theta_c.theta(index) feature neighborElement]
            elseif strcmp(obj.mode, 'useLocal')
                feature = mod((index - 1), size(obj.featureFunctions, 2) + size(obj.globalFeatureFunctions, 2)) + 1;
                Element = floor((index - 1)/(size(obj.featureFunctions, 2) + size(obj.globalFeatureFunctions, 2))) + 1;
                curr_theta = [obj.theta_c.theta(index) feature Element]
            elseif(strcmp(obj.mode, 'useLocalNeighbor') || strcmp(obj.mode, 'useLocalDiagNeighbor'))
                disp('theta feature coarseElement neighbor')
                curr_theta = [obj.theta_c.theta(index) obj.neighborDictionary(index, 1)...
                    obj.neighborDictionary(index, 2) obj.neighborDictionary(index, 3)]
            else
                curr_theta = [obj.theta_c.theta(index) index]
            end
            
            curr_sigma = obj.theta_c.Sigma
            mean_S = mean(obj.theta_cf.S)
            %curr_theta_hyperparam = obj.thetaPriorHyperparam
        end
        
        function obj = linearFilterUpdate(obj)
            if(obj.linFilt.totalUpdates > 0 && ~strcmp(obj.mode, 'useLocal'))
                error('Use local mode for sequential addition of basis functions')
            end
            
            if(obj.epoch > obj.linFilt.initialEpochs &&...
                    mod((obj.epoch - obj.linFilt.initialEpochs + 1), obj.linFilt.gap) == 0 &&...
                    obj.epoch ~= obj.epoch_old && obj.linFilt.updates < obj.linFilt.totalUpdates)
                obj.linFilt.updates = obj.linFilt.updates + 1;
                if strcmp(obj.linFilt.type, 'local')
                    obj = obj.addLinearFilterFeature;
                elseif strcmp(obj.linFilt.type, 'global')
                    obj = obj.addGlobalLinearFilterFeature;
                else
                    
                end
                %Recompute theta_c and sigma
                obj.theta_c = optTheta_c(obj.theta_c, obj.nTrain, obj.coarseScaleDomain.nEl, obj.XSqMean,...
                    obj.designMatrix, obj.XMean, obj.sigmaPriorType, obj.sigmaPriorHyperparam);
            end
        end
        
        function obj = predict(obj, mode)
            %Function to predict finescale output from generative model
            
            if(nargin < 2)
                mode = 'test';
            end
            
            %Load test file
            if strcmp(mode, 'self')
                %Self-prediction on training set
                Tf = obj.trainingDataMatfile.Tf(:, obj.nStart:(obj.nStart + obj.nTrain - 1));
            else
                Tf = obj.testDataMatfile.Tf(:, obj.testSamples);
            end
            obj = obj.loadTrainedParams;

            if strcmp(mode, 'self')
                obj = obj.computeDesignMatrix('train');
            else
                obj = obj.computeDesignMatrix('test');
            end
            
            %% Sample from p_c
            disp('Sampling from p_c...')
            if strcmp(mode, 'self')
                nTest = obj.nTrain;
            else
                nTest = numel(obj.testSamples);
            end
            Xsamples = zeros(obj.coarseScaleDomain.nEl, obj.nSamples_p_c, nTest);
            LambdaSamples{1} = zeros(obj.coarseScaleDomain.nEl, obj.nSamples_p_c);
            LambdaSamples = repmat(LambdaSamples, nTest, 1);
            meanEffCond = zeros(obj.coarseScaleDomain.nEl, nTest);

            for i = 1:nTest
                Xsamples(:, :, i) = mvnrnd(obj.designMatrix{i}*obj.theta_c.theta,...
                    obj.theta_c.Sigma, obj.nSamples_p_c)';
                LambdaSamples{i} = conductivityBackTransform(Xsamples(:, :, i), obj.conductivityTransformation);
                if(strcmp(obj.conductivityTransformation.type, 'log'))
                    meanEffCond(:, i) = exp(obj.designMatrix{i}*obj.theta_c.theta + .5*diag(obj.theta_c.Sigma));
                else
                    meanEffCond(:, i) = mean(LambdaSamples{i}, 2);
                end
            end
            disp('done')
            
            %% Run coarse model and sample from p_cf
            disp('Solving coarse model and sample from p_cf...')
            %             addpath('./heatFEM')
            TfMeanArray{1} = zeros(obj.fineScaleDomain.nNodes, 1);
            TfMeanArray = repmat(TfMeanArray, nTest, 1);
            TfVarArray = TfMeanArray;
            Tf_sq_mean = TfMeanArray;
            
            %To avoid broadcasting overhead
            nSamples = obj.nSamples_p_c;
            coarseDomain = obj.coarseScaleDomain;
            t_cf = obj.theta_cf;
            %             t_c = obj.theta_c;
            addpath('./heatFEM');
            parfor j = 1:nTest
                for i = 1:nSamples
                    D = zeros(2, 2, coarseDomain.nEl);
                    for e = 1:coarseDomain.nEl
                        D(:, :, e) = LambdaSamples{j}(e, i)*eye(2);
                    end
                    FEMout = heat2d(coarseDomain, D);
                    Tctemp = FEMout.Tff';
                    
                    %sample from p_cf
                    mu_cf = t_cf.mu + t_cf.W*Tctemp(:);
                    %only for diagonal S!!
                    %Sequentially compute mean and <Tf^2> to save memory
                    TfMeanArray{j} = ((i - 1)/i)*TfMeanArray{j} + (1/i)*mu_cf;  %U_f-integration can be done analyt.
                    Tf_sq_mean{j} = ((i - 1)/i)*Tf_sq_mean{j} + (1/i)*mu_cf.^2;
                end
                Tf_sq_mean{j} = Tf_sq_mean{j} + t_cf.S;
                Tf_var = abs(Tf_sq_mean{j} - TfMeanArray{j}.^2);  %abs to avoid negative variance due to numerical error
                meanTf_meanMCErr = mean(sqrt(Tf_var/nSamples))
                TfVarArray{j} = Tf_var;
                
                meanMahaErrTemp{j} = mean(sqrt((.5./(Tf_var)).*(Tf(:, j) - TfMeanArray{j}).^2));
                sqDist{j} = (Tf(:, j) - TfMeanArray{j}).^2;
                meanSqDistTemp{j} = mean(sqDist{j})
            end
            
            obj.meanPredMeanOutput = mean(cell2mat(TfMeanArray'), 2);
            obj.meanMahalanobisError = mean(cell2mat(meanMahaErrTemp));
            obj.meanSquaredDistance = mean(cell2mat(meanSqDistTemp));
            meanSqDistSq = mean(cell2mat(meanSqDistTemp).^2);
            obj.meanSquaredDistanceError = sqrt((meanSqDistSq - obj.meanSquaredDistance^2)/nTest);
            storeArray = false;
            if storeArray
                obj.predMeanArray = TfMeanArray;
                obj.predVarArray = TfVarArray;
            end
            
            plotPrediction = true;
            if plotPrediction
                f = figure('units','normalized','outerposition',[0 0 1 1]);
                pstart = 1;
                j = 1;
                max_Tf = max(max(Tf(:, pstart:(pstart + 5))));
                min_Tf = min(min(Tf(:, pstart:(pstart + 5))));
                if strcmp(mode, 'self')
                    cond = obj.trainingDataMatfile.cond(:, obj.nStart:(obj.nStart + obj.nTrain - 1));
                    cond = cond(:, pstart:(pstart + 5));
                else
                    cond = obj.testDataMatfile.cond(:, pstart:(pstart + 5));
                end
                %to use same color scale
                cond = ((min_Tf - max_Tf)/(min(min(cond)) - max(max(cond))))*cond + max_Tf - ...
                    ((min_Tf - max_Tf)/(min(min(cond)) - max(max(cond))))*max(max(cond));
                for i = pstart:(pstart + 5)
                    subplot(2, 3, j)
                    s(j, 1) = surf(reshape(Tf(:, i), (obj.nElFX + 1), (obj.nElFY + 1)));
                    s(j, 1).LineStyle = 'none';
                    hold on;
                    s(j, 2) = surf(reshape(TfMeanArray{i}, (obj.nElFX + 1), (obj.nElFY + 1)));
                    s(j, 2).LineStyle = 'none';
                    s(j, 2).FaceColor = 'b';
                    s(j, 3) = surf(reshape(TfMeanArray{i}, (obj.nElFX + 1), (obj.nElFY + 1)) +...
                        sqrt(reshape(TfVarArray{i}, (obj.nElFX + 1), (obj.nElFY + 1))));
                    s(j, 3).LineStyle = 'none';
                    s(j, 3).FaceColor = [.85 .85 .85];
                    s(j, 4) = surf(reshape(TfMeanArray{i}, (obj.nElFX + 1), (obj.nElFY + 1)) -...
                        sqrt(reshape(TfVarArray{i}, (obj.nElFX + 1), (obj.nElFY + 1))));
                    s(j, 4).LineStyle = 'none';
                    s(j, 4).FaceColor = [.85 .85 .85];
                    ax = gca;
                    ax.FontSize = 30;
                    im(j) = imagesc(reshape(cond(:, i), obj.fineScaleDomain.nElX, obj.fineScaleDomain.nElY));
                    xticks([0 64 128 192 256]);
                    yticks([0 64 128 192 256]);
                    zticks(100:100:800)
                    xticklabels({});
                    yticklabels({});
                    zticklabels({});
                    axis tight;
                    axis square;
                    box on;
                    view(-60, 15)
                    zlim([0 800]);
                    caxis([min_Tf max_Tf]);
                    j = j + 1;
                end
                print(f, './predictions', '-dpng', '-r300')
            end
        end

        
        
        %% Design matrix functions
        function obj = getCoarseElement(obj)
            debug = false;
            obj.E = zeros(obj.fineScaleDomain.nEl, 1);
            e = 1;  %element number
            for row_fine = 1:obj.fineScaleDomain.nElY
                %coordinate of lower boundary of fine element
                y_coord = obj.fineScaleDomain.cum_lElY(row_fine);
                row_coarse = sum(y_coord >= obj.coarseScaleDomain.cum_lElY);
                for col_fine = 1:obj.fineScaleDomain.nElX
                    %coordinate of left boundary of fine element
                    x_coord = obj.fineScaleDomain.cum_lElX(col_fine);
                    col_coarse = sum(x_coord >= obj.coarseScaleDomain.cum_lElX);
                    obj.E(e) = (row_coarse - 1)*obj.coarseScaleDomain.nElX + col_coarse;
                    e = e + 1;
                end
            end
            
            obj.E = reshape(obj.E, obj.fineScaleDomain.nElX, obj.fineScaleDomain.nElY);
            if debug
                figure
                imagesc(obj.E)
                pause
            end
        end

        function [lambdak, xk] = get_coarseElementConductivities(obj, mode)
            %load finescale conductivity field
            if strcmp(mode, 'train')
                conductivity = obj.trainingDataMatfile.cond(:, obj.trainingSamples);
            elseif strcmp(mode, 'test')
                conductivity = obj.testDataMatfile.cond(:, obj.testSamples);
            else
                error('Either train or test mode')
            end
            nData = size(conductivity, 2);
            %Mapping from fine cell index to coarse cell index
            obj = obj.getCoarseElement;
                        
            %Open parallel pool
            %addpath('./computation')
            %parPoolInit(nTrain);
            EHold = obj.E;  %this is for parfor efficiency
            
            %prealloc
            lambdak = cell(nData, obj.coarseScaleDomain.nEl);
            if(nargout > 1)
                xk = lambdak;
            end
            for s = 1:nData
                %inputs belonging to same coarse element are in the same column of xk. They are ordered in
                %x-direction.
                %Get conductivity fields in coarse cell windows
                %Might be wrong for non-square fine scale domains
                conductivityMat = reshape(conductivity(:, s), obj.fineScaleDomain.nElX,...
                    obj.fineScaleDomain.nElY);
                for e = 1:obj.coarseScaleDomain.nEl
                    indexMat = (EHold == e);
                    lambdakTemp = conductivityMat.*indexMat;
                    %Cut elements from matrix that do not belong to coarse cell
                    lambdakTemp(~any(lambdakTemp, 2), :) = [];
                    lambdakTemp(:, ~any(lambdakTemp, 1)) = [];
                    lambdak{s, e} = lambdakTemp;
                    if(nargout > 1)
                        xk{s, e} = conductivityTransform(lambdak{s, e}, obj.conductivityTransformation);
                    end
                end
            end
        end
        
        function obj = computeKernelMatrix(obj, mode)
            %Must be called BEFORE setting up any local lode (useLocal, useNeighbor,...)
            %It is instructive to rescale/standardize the design matrix before using kernels
            disp('Using kernel regression mode. It is highly recommended to use the RVM prior!')
            
            %check if coarse mesh is square
            if(all(obj.coarseGridVectorX == obj.coarseGridVectorX(1)) &&...
                    all(obj.coarseGridVectorY == obj.coarseGridVectorX(1)))
                %Coarse model is a square grid
                isSquare = true;
            else
                isSquare = false;
            end
            
            if isSquare
                %We take all kernels together from all macro-cells
                %prealloc
                obj.kernelMatrix{1} = zeros(obj.coarseScaleDomain.nEl, obj.coarseScaleDomain.nEl*obj.nTrain);
                obj.kernelMatrix = repmat(obj.kernelMatrix, obj.nTrain, 1);
                
                %Fill kernelMatrix - can this be done more efficiently?
                if strcmp(mode, 'train')
                    for n = 1:obj.nTrain
                        for k = 1:obj.coarseScaleDomain.nEl
                            f = 1;
                            for nn = 1:obj.nTrain
                                for kk = 1:obj.coarseScaleDomain.nEl
                                    %kernelDiff is a row vector
                                    kernelDiff = obj.designMatrix{n}(k, :) - obj.designMatrix{nn}(kk, :);
                                    obj.kernelMatrix{n}(k, f) = obj.kernelFunction(kernelDiff);
                                    f = f + 1;
                                end
                            end
                        end
                    end
                elseif strcmp(mode, 'test')
                    disp('Computing kernel matrix in test mode. Make sure to load correct training design matrix!')
                    load('./persistentData/trainDesignMatrix.mat');
                    for n = 1:numel(obj.designMatrix)
                        for k = 1:obj.coarseScaleDomain.nEl
                            f = 1;
                            for nn = 1:obj.nTrain
                                for kk = 1:obj.coarseScaleDomain.nEl
                                    %kernelDiff is a row vector
                                    kernelDiff = obj.designMatrix{n}(k, :) - designMatrix{nn}(kk, :);
                                    obj.kernelMatrix{n}(k, f) = obj.kernelFunction(kernelDiff);
                                    f = f + 1;
                                end
                            end
                        end
                    end
                else
                    error('Use either train or test mode')
                end
            else
                error('Non-square meshes not yet available')
            end
            if(strcmp(mode, 'train') && length(obj.theta_c.theta) ~= obj.coarseScaleDomain.nEl*obj.nTrain)
                disp('Setting dimension of theta_c right, initializing at 0')
                obj.theta_c.theta = zeros(obj.coarseScaleDomain.nEl*obj.nTrain, 1);
            end
        end

        function obj = computeDesignMatrix(obj, mode, recompute)
            %Actual computation of design matrix
            %set recompute to true if design matrices have to be recomputed during optimization (parametric features)
            tic
            
            if(obj.loadDesignMatrix && ~recompute)
                load(strcat('./persistentData/', mode, 'DesignMatrix.mat'));
                obj.designMatrix = designMatrix;
                if obj.useAutoEnc
                    load('./persistentData/latentDim');
                    obj.latentDim = latentDim;
                end
                if(obj.linFilt.totalUpdates > 0)
                    load('./persistentData/lambdak');
                    obj.lambdak = lambdak;
                    obj.xk = xk;
                end
            else
                debug = false; %for debug mode
                disp('Compute design matrices...')
                
                if strcmp(mode, 'train')
                    dataFile = obj.trainingDataMatfile;
                    dataSamples = obj.trainingSamples;
                elseif strcmp(mode, 'test')
                    dataFile = obj.testDataMatfile;
                    dataSamples = obj.testSamples;
                else
                    error('Compute design matrices for train or test data?')
                end
                nData = numel(dataSamples);
                
                %load finescale conductivity field
                conductivity = dataFile.cond(:, dataSamples);
                conductivity = num2cell(conductivity, 1);   %to avoid parallelization communication overhead
                nFeatureFunctions = size(obj.featureFunctions, 2);
                nGlobalFeatureFunctions = size(obj.globalFeatureFunctions, 2);
                phi = obj.featureFunctions;
                phiGlobal = obj.globalFeatureFunctions;
                
                %Open parallel pool
                %addpath('./computation')
                %parPoolInit(nData);
                PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, nFeatureFunctions + nGlobalFeatureFunctions);
                PhiCell = repmat(PhiCell, nData, 1);
                [lambdak, xk] = obj.get_coarseElementConductivities(mode);
                if(obj.linFilt.totalUpdates > 0)
                    %These only need to be stored if we sequentially add features
                    obj.lambdak = lambdak;
                    obj.xk = xk;
                    save('./persistentData/lambdak', 'lambdak', 'xk');
                end
                
                if obj.useAutoEnc
                    %should work for training as well as testing
                    %Only for square grids!!!
                    lambdakMat = zeros(numel(lambdak{1}), numel(lambdak));
                    m = 1;
                    for n = 1:size(lambdak, 1)
                        for k = 1:size(lambdak, 2)
                            lambdakMat(:, m) = lambdak{n, k}(:);
                            m = m + 1;
                        end
                    end
                    lambdakMatBin = logical(lambdakMat - obj.lowerConductivity);
                    %Encoded version of test samples
                    load('./autoencoder/trainedAutoencoder.mat');
                    latentMu = ba.encode(lambdakMatBin);
                    obj.latentDim = ba.latentDim;
                    latentDim = ba.latentDim;
                    save('./persistentData/latentDim', 'latentDim');
                    if ~debug
                        clear ba;
                    end
                    latentMu = reshape(latentMu, obj.latentDim, obj.coarseScaleDomain.nEl, nData);
                end
                
                
                for s = 1:nData    %for very cheap features, serial evaluation might be more efficient
                    %inputs belonging to same coarse element are in the same column of xk. They are ordered in
                    %x-direction.
                    
                    %construct design matrix
                    for i = 1:obj.coarseScaleDomain.nEl
                        %local features
                        for j = 1:nFeatureFunctions
                            %only take pixels of corresponding macro-cell as input for features
                            PhiCell{s}(i, j) = phi{i, j}(lambdak{s, i});
                        end
                        %global features
                        for j = 1:nGlobalFeatureFunctions
                            %Take whole microstructure as input for feature function
                            %Might be wrong for non-square fine scale domains
                            conductivityMat = reshape(conductivity(:, s), obj.fineScaleDomain.nElX,...
                                obj.fineScaleDomain.nElY);
                            PhiCell{s}(i, nFeatureFunctions + j) = phiGlobal{i, j}(conductivityMat);
                        end
                        if obj.useAutoEnc
                            for j = 1:obj.latentDim
                                PhiCell{s}(i, nFeatureFunctions + nGlobalFeatureFunctions + j) = latentMu(j, i, s);
                            end
                        end
                    end
                end
                
                if debug
                    for n = 1:nData
                        for k = 1:obj.coarseScaleDomain.nEl
                            decodedDataTest = ba.decode(latentMu(:, k, n));
                            subplot(1,3,1)
                            imagesc(reshape(decodedDataTest, 64, 64))
                            axis square
                            grid off
                            yticks({})
                            xticks({})
                            colorbar
                            subplot(1,3,2)
                            imagesc(reshape(decodedDataTest > 0.5, 64, 64))
                            axis square
                            grid off
                            yticks({})
                            xticks({})
                            colorbar
                            subplot(1,3,3)
                            imagesc(lambdak{n, k})
                            axis square
                            yticks({})
                            xticks({})
                            grid off
                            colorbar
                            drawnow
                            pause(.5)
                        end
                    end
                end
                obj.designMatrix = PhiCell;
                %Check for real finite inputs
                for i = 1:nData
                    if(~all(all(all(isfinite(obj.designMatrix{i})))))
                        dataPoint = i
                        [coarseElement, featureFunction] = ind2sub(size(obj.designMatrix{i}),...
                            find(~isfinite(obj.designMatrix{i})))
                        warning('Non-finite design matrix. Setting non-finite component to 0.')
                        obj.designMatrix{i}(~isfinite(obj.designMatrix{i})) = 0;
                    elseif(~all(all(all(isreal(obj.designMatrix{i})))))
                        warning('Complex feature function output:')
                        dataPoint = i
                        [coarseElement, featureFunction] = ind2sub(size(obj.designMatrix{i}),...
                            find(imag(obj.designMatrix{i})))
                        disp('Ignoring imaginary part...')
                        obj.designMatrix{i} = real(obj.designMatrix{i});
                    end
                end
                disp('done')
                %Include second order combinations of features
                obj = obj.secondOrderFeatures(mode);
                %Normalize design matrices
                if strcmp(obj.featureScaling, 'standardize')
                    obj = obj.standardizeDesignMatrix(mode);
                elseif strcmp(obj.featureScaling, 'rescale')
                    obj = obj.rescaleDesignMatrix(mode);
                elseif strcmp(obj.featureScaling, 'normalize')
                    obj = obj.normalizeDesignMatrix(mode);
                else
                    disp('No feature scaling used...')
                end
                
                if(obj.useKernels)
                    if strcmp(obj.bandwidthSelection, 'silverman')
                        %Compute feature function variances
                        if strcmp(mode, 'train')
                            if isempty(obj.featureFunctionMean)
                                obj = obj.computeFeatureFunctionMean;
                            end
                            if isempty(obj.featureFunctionSqMean)
                                obj = obj.computeFeatureFunctionSqMean;
                            end
                            featureFunctionStd = sqrt(obj.featureFunctionSqMean - obj.featureFunctionMean.^2);
                            nFeatures = numel(obj.featureFunctionMean);
                            %Silverman's rule of thumb
                            if strcmp(obj.mode, 'none')
                                obj.kernelBandwidth = (obj.nTrain*obj.coarseScaleDomain.nEl)^(-1/(nFeatures + 4))*...
                                    featureFunctionStd;
                            elseif strcmp(obj.mode, 'useLocal')
                                obj.kernelBandwidth = obj.nTrain^(-1/(nFeatures + 4))*featureFunctionStd;
                            else
                                error('No rule of thumb implemented for this mode:')
                                obj.mode
                            end
                            kernelBandwidth = obj.kernelBandwidth;
                            save('./persistentData/kernelBandwidth.mat', 'kernelBandwidth');
                        elseif strcmp(mode, 'test')
                            load('./persistentData/kernelBandwidth.mat');
                            obj.kernelBandwidth = kernelBandwidth;
                        else
                            error('Choose test or train mode!')
                        end
                    elseif strcmp(obj.bandwidthSelection, 'fixed')
                        %change nothing
                    else
                        error('Invalid bandwidth selection')
                    end
                    obj = obj.computeKernelMatrix(mode);
                end
                
                %Design matrix is always stored in its original form. Local modes are applied after
                %loading
                designMatrix = obj.designMatrix;
                save(strcat('./persistentData/', mode, 'DesignMatrix.mat'), 'designMatrix')
            end
            if obj.useKernels
                %This step might be confusing: after storing, we replace the design matrix by the kernel matrix
                obj.designMatrix = obj.kernelMatrix;
                obj.kernelMatrix = [];  %to save memory
            end
            %Use specific nonlocality mode
            if strcmp(obj.mode, 'useNeighbor')
                %use feature function information from nearest neighbors
                obj = obj.includeNearestNeighborFeatures;
            elseif strcmp(obj.mode, 'useLocalNeighbor')
                obj = obj.includeLocalNearestNeighborFeatures;
            elseif strcmp(obj.mode, 'useLocalDiagNeighbor')
                obj = obj.includeLocalDiagNeighborFeatures;
            elseif strcmp(obj.mode, 'useDiagNeighbor')
                %use feature function information from nearest and diagonal neighbors
                obj = obj.includeDiagNeighborFeatures;
            elseif strcmp(obj.mode, 'useLocal')
                %Use separate parameters for every macro-cell
                obj = obj.localTheta_c;
            end
            obj = obj.computeSumPhiTPhi;
            Phi_computation_time = toc
        end
        
        function obj = secondOrderFeatures(obj, mode)
            %Includes second order multinomial terms, i.e. a_ij phi_i phi_j, where a_ij is logical.
            %Squared term phi_i^2 if a_ii ~= 0. To be executed directly after feature function
            %computation.
            
            assert(all(all(islogical(obj.secondOrderTerms))), 'A must be a logical array of nFeatures x nFeatures')
            %Consider every term only once
            assert(sum(sum(tril(obj.secondOrderTerms, -1))) == 0, 'Matrix A must be upper triangular')
            
            nFeatureFunctions = size(obj.featureFunctions, 2) + size(obj.globalFeatureFunctions, 2)...
                + obj.latentDim;
            nSecondOrderTerms = sum(sum(obj.secondOrderTerms));
            if nSecondOrderTerms
                disp('Using second order terms of feature functions...')
            end
            PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, nSecondOrderTerms + nFeatureFunctions);
            if strcmp(mode, 'train')
                nData = obj.nTrain;
            elseif strcmp(mode, 'test')
                nData = numel(obj.testSamples);
            end
            PhiCell = repmat(PhiCell, nData, 1);
            
            for s = 1:nData
                %The first columns contain first order terms
                PhiCell{s}(:, 1:nFeatureFunctions) = obj.designMatrix{s};
                
                %Second order terms
                f = 1;
                for r = 1:size(obj.secondOrderTerms, 1)
                    for c = r:size(obj.secondOrderTerms, 2)
                        if obj.secondOrderTerms(r, c)
                            PhiCell{s}(:, nFeatureFunctions + f) = ...
                                PhiCell{s}(:, r).*PhiCell{s}(:, c);
                            f = f + 1;
                        end
                    end
                end
            end
            obj.designMatrix = PhiCell;
            disp('done')
        end%secondOrderFeatures
        
        function obj = computeFeatureFunctionMean(obj)
            %Must be executed BEFORE useLocal etc.
            obj.featureFunctionMean = 0;
            for n = 1:numel(obj.designMatrix)
                obj.featureFunctionMean = obj.featureFunctionMean + mean(obj.designMatrix{n}, 1);
            end
            obj.featureFunctionMean = obj.featureFunctionMean/numel(obj.designMatrix);
        end

        function obj = computeFeatureFunctionSqMean(obj)
            featureFunctionSqSum = 0;
            for i = 1:numel(obj.designMatrix)
                featureFunctionSqSum = featureFunctionSqSum + sum(obj.designMatrix{i}.^2, 1);
            end
            obj.featureFunctionSqMean = featureFunctionSqSum/...
                (numel(obj.designMatrix)*size(obj.designMatrix{1}, 1));
        end

        function obj = standardizeDesignMatrix(obj, mode)
            %Standardize covariates to have 0 mean and unit variance
            disp('Standardize design matrix...')
            %Compute std
            if strcmp(mode, 'test')
                featureFunctionStd = sqrt(obj.featureFunctionSqMean - obj.featureFunctionMean.^2);
            else
                obj = obj.computeFeatureFunctionMean;
                obj = obj.computeFeatureFunctionSqMean;
                featureFunctionStd = sqrt(obj.featureFunctionSqMean - obj.featureFunctionMean.^2);
                if(any(~isreal(featureFunctionStd)))
                    warning('Imaginary standard deviation. Setting it to 0.')
                    featureFunctionStd = real(featureFunctionStd);
                end
            end
            
            %centralize
            for i = 1:numel(obj.designMatrix)
                obj.designMatrix{i} = obj.designMatrix{i} - obj.featureFunctionMean;
            end
            
            %normalize
            for i = 1:numel(obj.designMatrix)
                obj.designMatrix{i} = obj.designMatrix{i}./featureFunctionStd;
            end
            
            %Check for finiteness
            for i = 1:numel(obj.designMatrix)
                if(~all(all(all(isfinite(obj.designMatrix{i})))))
                    warning('Non-finite design matrix. Setting non-finite component to 0.')
                    obj.designMatrix{i}(~isfinite(obj.designMatrix{i})) = 0;
                elseif(~all(all(all(isreal(obj.designMatrix{i})))))
                    warning('Complex feature function output:')
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(obj.designMatrix{i}),...
                        find(imag(obj.designMatrix{i})))
                    disp('Ignoring imaginary part...')
                    obj.designMatrix{i} = real(obj.designMatrix{i});
                end
            end
            obj.saveNormalization('standardization');
            disp('done')
        end
        
        function obj = normalizeDesignMatrix(obj, mode)
            %Standardize covariates to have 0 mean and unit variance
            disp('Normalize design matrix...')
            %Compute std
            if strcmp(mode, 'test')
                featureFunctionStd = sqrt(obj.featureFunctionSqMean - obj.featureFunctionMean.^2);
            else
                obj = obj.computeFeatureFunctionMean;
                obj = obj.computeFeatureFunctionSqMean;
                featureFunctionStd = sqrt(obj.featureFunctionSqMean - obj.featureFunctionMean.^2);
                if(any(~isreal(featureFunctionStd)))
                    warning('Imaginary standard deviation. Setting it to 0.')
                    featureFunctionStd = real(featureFunctionStd);
                end
            end
            
            %normalize
            for i = 1:numel(obj.designMatrix)
                obj.designMatrix{i} = obj.designMatrix{i}./featureFunctionStd;
            end
            
            %Check for finiteness
            for i = 1:numel(obj.designMatrix)
                if(~all(all(all(isfinite(obj.designMatrix{i})))))
                    warning('Non-finite design matrix. Setting non-finite component to 0.')
                    obj.designMatrix{i}(~isfinite(obj.designMatrix{i})) = 0;
                elseif(~all(all(all(isreal(obj.designMatrix{i})))))
                    warning('Complex feature function output:')
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(obj.designMatrix{i}),...
                        find(imag(obj.designMatrix{i})))
                    disp('Ignoring imaginary part...')
                    obj.designMatrix{i} = real(obj.designMatrix{i});
                end
            end
            obj.saveNormalization('standardization');
            disp('done')
        end
        
        function obj = computeFeatureFunctionMinMax(obj)
            %Computes min/max of feature function outputs over training data, separately for every
            %macro cell
            obj.featureFunctionMin = obj.designMatrix{1};
            obj.featureFunctionMax = obj.designMatrix{1};
            for n = 1:numel(obj.designMatrix)
                obj.featureFunctionMin(obj.featureFunctionMin > obj.designMatrix{n}) =...
                    obj.designMatrix{n}(obj.featureFunctionMin > obj.designMatrix{n});
                obj.featureFunctionMax(obj.featureFunctionMax < obj.designMatrix{n}) =...
                    obj.designMatrix{n}(obj.featureFunctionMax < obj.designMatrix{n});
            end
        end
        
        function obj = rescaleDesignMatrix(obj, mode)
            %Rescale design matrix s.t. outputs are between 0 and 1
            disp('Rescale design matrix...')
            if strcmp(mode, 'test')
                featFuncDiff = obj.featureFunctionMax - obj.featureFunctionMin;
                %to avoid irregularities due to rescaling (if every macro cell has the same feature function output)
                obj.featureFunctionMin(featFuncDiff == 0) = 0;
                featFuncDiff(featFuncDiff == 0) = 1;
                for n = 1:numel(obj.designMatrix)
                    obj.designMatrix{n} = (obj.designMatrix{n} - obj.featureFunctionMin)./(featFuncDiff);
                end
            else
                obj = obj.computeFeatureFunctionMinMax;
                featFuncDiff = obj.featureFunctionMax - obj.featureFunctionMin;
                %to avoid irregularities due to rescaling (if every macro cell has the same feature function output)
                obj.featureFunctionMin(featFuncDiff == 0) = 0;
                featFuncDiff(featFuncDiff == 0) = 1;
                for n = 1:numel(obj.designMatrix)
                    obj.designMatrix{n} = (obj.designMatrix{n} - obj.featureFunctionMin)./(featFuncDiff);
                end
            end
            %Check for finiteness
            for n = 1:numel(obj.designMatrix)
                if(~all(all(all(isfinite(obj.designMatrix{n})))))
                    warning('Non-finite design matrix. Setting non-finite component to 0.')
                    obj.designMatrix{n}(~isfinite(obj.designMatrix{n})) = 0;
                    dataPoint = n
                    [coarseElement, featureFunction] = ind2sub(size(obj.designMatrix{n}),...
                        find(~isfinite(obj.designMatrix{n})))
                elseif(~all(all(all(isreal(obj.designMatrix{n})))))
                    warning('Complex feature function output:')
                    dataPoint = n
                    [coarseElement, featureFunction] = ind2sub(size(obj.designMatrix{n}),...
                        find(imag(obj.designMatrix{n})))
                    disp('Ignoring imaginary part...')
                    obj.designMatrix{n} = real(obj.designMatrix{n});
                end
            end
            obj.saveNormalization('rescaling');
            disp('done')
        end
        
        function saveNormalization(obj, type)
            disp('Saving design matrix normalization...')
            if(isempty(obj.featureFunctionMean))
                obj = obj.computeFeatureFunctionMean;
            end
            if(isempty(obj.featureFunctionSqMean))
                obj = obj.computeFeatureFunctionSqMean;
            end
            if ~exist('./data')
                mkdir('./data');
            end
            if strcmp(type, 'standardization')
                featureFunctionMean = obj.featureFunctionMean;
                featureFunctionSqMean = obj.featureFunctionSqMean;
                save('./data/featureFunctionMean', 'featureFunctionMean', '-ascii');
                save('./data/featureFunctionSqMean', 'featureFunctionSqMean', '-ascii');
            elseif strcmp(type, 'rescaling')
                featureFunctionMin = obj.featureFunctionMin;
                featureFunctionMax = obj.featureFunctionMax;
                save('./data/featureFunctionMin', 'featureFunctionMin', '-ascii');
                save('./data/featureFunctionMax', 'featureFunctionMax', '-ascii');
            else
                error('Which type of data normalization?')
            end
        end
        
        function obj = includeNearestNeighborFeatures(obj)
            %Includes feature function information of neighboring cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest neighbor feature function information...')
            nFeatureFunctionsTotal = size(obj.designMatrix{1}, 2);
            PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, 5*nFeatureFunctionsTotal);
            nData = numel(obj.designMatrix);
            PhiCell = repmat(PhiCell, nData, 1);
            
            for n = 1:nData
                %The first columns contain feature function information of the original cell
                PhiCell{n}(:, 1:nFeatureFunctionsTotal) = obj.designMatrix{n};
                
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                for k = 1:obj.coarseScaleDomain.nEl
                    if(mod(k, obj.coarseScaleDomain.nElX) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{n}(k, (nFeatureFunctionsTotal + 1):(2*nFeatureFunctionsTotal)) =...
                           obj.designMatrix{n}(k + 1, :);
                    end
                    
                    if(k <= obj.coarseScaleDomain.nElX*(obj.coarseScaleDomain.nElY - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{n}(k, (2*nFeatureFunctionsTotal + 1):(3*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(k + obj.coarseScaleDomain.nElX, :);
                    end
                    
                    if(mod(k - 1, obj.coarseScaleDomain.nElX) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{n}(k, (3*nFeatureFunctionsTotal + 1):(4*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(k - 1, :);
                    end
                    
                    if(k > obj.coarseScaleDomain.nElX)
                        %lower neighbor of coarse element exists
                        PhiCell{n}(k, (4*nFeatureFunctionsTotal + 1):(5*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(k - obj.coarseScaleDomain.nElX, :);
                    end
                end
            end
            obj.designMatrix = PhiCell;
            disp('done')
        end%includeNearestNeighborFeatures
        
        function obj = includeLocalNearestNeighborFeatures(obj)
            %Includes feature function information of neighboring cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest neighbor feature function information separately for each cell...')
            nFeatureFunctionsTotal = size(obj.designMatrix{1}, 2);
            PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, 5*nFeatureFunctionsTotal);
            nData = numel(obj.designMatrix);
            PhiCell = repmat(PhiCell, nData, 1);
            
            for n = 1:nData
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                k = 0;
                for i = 1:obj.coarseScaleDomain.nEl
                    PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                        obj.designMatrix{n}(i, :);
                    obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                        (1:nFeatureFunctionsTotal)'; %feature index
                    obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                        i; %coarse element index
                    obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                        0; %center element
                    k = k + 1;
                    if(mod(i, obj.coarseScaleDomain.nElX) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i + 1, :);
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            1; %right neighbor
                        k = k + 1;
                    end
                    
                    if(i <= obj.coarseScaleDomain.nElX*(obj.coarseScaleDomain.nElY - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i + obj.coarseScaleDomain.nElX, :);
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            2; %upper neighbor
                        k = k + 1;
                    end
                    
                    if(mod(i - 1, obj.coarseScaleDomain.nElX) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i - 1, :);
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            3; %left neighbor
                        k = k + 1;
                    end
                    
                    if(i > obj.coarseScaleDomain.nElX)
                        %lower neighbor of coarse element exists
                        PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i - obj.coarseScaleDomain.nElX, :);
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            4; %lower neighbor
                        k = k + 1;
                    end
                end
            end
            obj.designMatrix = PhiCell;
            disp('done')
        end%includeLocalNearestNeighborFeatures
        
        function obj = includeDiagNeighborFeatures(obj)
            %includes feature function information of all other cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest and diagonal neighbor feature function information...')
            nFeatureFunctionsTotal = size(obj.designMatrix{1}, 2);
            PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, 9*nFeatureFunctionsTotal);
            nData = numel(obj.designMatrix);
            PhiCell = repmat(PhiCell, nData, 1);
            
            for n = 1:nData
                %The first columns contain feature function information of the original cell
                PhiCell{n}(:, 1:nFeatureFunctionsTotal) = obj.designMatrix{n};
                
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                for i = 1:obj.coarseScaleDomain.nEl
                    if(mod(i, obj.coarseScaleDomain.nElX) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{n}(i, (nFeatureFunctionsTotal + 1):(2*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i + 1, :);
                        if(i <= obj.coarseScaleDomain.nElX*(obj.coarseScaleDomain.nElY - 1))
                            %upper right neighbor of coarse element exists
                            PhiCell{n}(i, (2*nFeatureFunctionsTotal + 1):(3*nFeatureFunctionsTotal)) =...
                                obj.designMatrix{n}(i + obj.coarseScaleDomain.nElX + 1, :);
                        end
                    end
                    
                    if(i <= obj.coarseScaleDomain.nElX*(obj.coarseScaleDomain.nElY - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{n}(i, (3*nFeatureFunctionsTotal + 1):(4*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i + obj.coarseScaleDomain.nElX, :);
                        if(mod(i - 1, obj.coarseScaleDomain.nElX) ~= 0)
                            %upper left neighbor exists
                            PhiCell{n}(i, (4*nFeatureFunctionsTotal + 1):(5*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i + obj.coarseScaleDomain.nElX - 1, :);
                        end
                    end
                    
                    if(mod(i - 1, obj.coarseScaleDomain.nElX) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{n}(i, (5*nFeatureFunctionsTotal + 1):(6*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i - 1, :);
                        if(i > obj.coarseScaleDomain.nElX)
                            %lower left neighbor exists
                            PhiCell{n}(i, (6*nFeatureFunctionsTotal + 1):(7*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i - obj.coarseScaleDomain.nElX - 1, :);
                        end
                    end
                    
                    if(i > obj.coarseScaleDomain.nElX)
                        %lower neighbor of coarse element exists
                        PhiCell{n}(i, (7*nFeatureFunctionsTotal + 1):(8*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i - obj.coarseScaleDomain.nElX, :);
                        if(mod(i, obj.coarseScaleDomain.nElX) ~= 0)
                            %lower right neighbor exists
                            PhiCell{n}(i, (8*nFeatureFunctionsTotal + 1):(9*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i - obj.coarseScaleDomain.nElX + 1, :);
                        end
                    end
                end
            end
            obj.designMatrix = PhiCell;
            disp('done')
        end%includeDiagNeighborFeatures

        function obj = includeLocalDiagNeighborFeatures(obj)
            %Includes feature function information of direct and diagonal neighboring cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest + diagonal neighbor feature function information separately for each cell...')
            nFeatureFunctionsTotal = size(obj.designMatrix{1}, 2);
            nData = numel(obj.designMatrix);
%             PhiCell = repmat(PhiCell, nTrain, 1);
            
            for n = 1:nData
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                k = 0;
                for i = 1:obj.coarseScaleDomain.nEl
                    PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                        obj.designMatrix{n}(i, :);
                    obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                        (1:nFeatureFunctionsTotal)'; %feature index
                    obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                        i; %coarse element index
                    obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                        0; %center element
                    k = k + 1;
                    if(mod(i, obj.coarseScaleDomain.nElX) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i + 1, :);
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            1; %right neighbor
                        k = k + 1;
                        
                        if(i <= obj.coarseScaleDomain.nElX*(obj.coarseScaleDomain.nElY - 1))
                            %upper right neighbor of coarse element exists
                            PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                                obj.designMatrix{n}(i + obj.coarseScaleDomain.nElX + 1, :);
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                                (1:nFeatureFunctionsTotal)'; %feature index
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                                i; %coarse element index
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                                2; % upper right neighbor
                            k = k + 1;
                        end
                        
                    end
                    
                    
                    if(i <= obj.coarseScaleDomain.nElX*(obj.coarseScaleDomain.nElY - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i + obj.coarseScaleDomain.nElX, :);
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            2; %upper neighbor
                        k = k + 1;
                        
                        if(mod(i - 1, obj.coarseScaleDomain.nElX) ~= 0)
                            %upper left neighbor of coarse element exists
                            PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                                obj.designMatrix{n}(i + obj.coarseScaleDomain.nElX - 1, :);
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                                (1:nFeatureFunctionsTotal)'; %feature index
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                                i; %coarse element index
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                                4; % upper left neighbor
                            k = k + 1;
                        end
                        
                    end
                    
                    
                    if(mod(i - 1, obj.coarseScaleDomain.nElX) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i - 1, :);
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            3; %left neighbor
                        k = k + 1;
                        
                        if(i > obj.coarseScaleDomain.nElX)
                            %lower left neighbor of coarse element exists
                            PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                                obj.designMatrix{n}(i - obj.coarseScaleDomain.nElX - 1, :);
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                                (1:nFeatureFunctionsTotal)'; %feature index
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                                i; %coarse element index
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                                6; % lower left neighbor
                            k = k + 1;
                        end
                        
                    end
                    
                    
                    if(i > obj.coarseScaleDomain.nElX)
                        %lower neighbor of coarse element exists
                        PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            obj.designMatrix{n}(i - obj.coarseScaleDomain.nElX, :);
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            4; %lower neighbor
                        k = k + 1;
                        
                        if(mod(i, obj.coarseScaleDomain.nElX) ~= 0)
                            %lower right neighbor of coarse element exists
                            PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                                obj.designMatrix{n}(i - obj.coarseScaleDomain.nElX + 1, :);
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                                (1:nFeatureFunctionsTotal)'; %feature index
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                                i; %coarse element index
                            obj.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                                8; % lower right neighbor
                            k = k + 1;
                        end
                        
                    end
                end
            end
            obj.designMatrix = PhiCell;
            disp('done')
        end%includeLocalDiagNeighborFeatures

        function obj = localTheta_c(obj)
            %Sets separate coefficients theta_c for each macro-cell in a single microstructure
            %sample
            %Can never be executed before rescaling/standardization of design Matrix!
            debug = false; %debug mode
            disp('Using separate feature coefficients theta_c for each macro-cell in a microstructure...')
            nFeatureFunctionsTotal = size(obj.designMatrix{1}, 2);
            PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, obj.coarseScaleDomain.nEl*nFeatureFunctionsTotal);
            nData = numel(obj.designMatrix);
            PhiCell = repmat(PhiCell, nData, 1);
            
            %Reassemble design matrix
            for n = 1:nData
                for i = 1:obj.coarseScaleDomain.nEl
                    PhiCell{n}(i, ((i - 1)*nFeatureFunctionsTotal + 1):(i*nFeatureFunctionsTotal)) = ...
                        obj.designMatrix{n}(i, :);
                end
                PhiCell{n} = sparse(PhiCell{n});
            end
            if debug
                firstDesignMatrixBeforeLocal = obj.designMatrix{1}
                firstDesignMatrixAfterLocal = full(PhiCell{1})
                pause
            end
            obj.designMatrix = PhiCell;
            disp('done')
        end%localTheta_c
        
        function obj = computeSumPhiTPhi(obj)
            obj.sumPhiTPhi = 0;
            for n = 1:numel(obj.designMatrix)
                obj.sumPhiTPhi = obj.sumPhiTPhi + obj.designMatrix{n}'*obj.designMatrix{n};
            end
            if strcmp(obj.mode, 'useLocal')
                obj.sumPhiTPhi = sparse(obj.sumPhiTPhi);
            end
        end
        
        function k = kernelFunction(obj, kernelDiff)
            %kernelDiff is the difference of the dependent variable to the kernel center
            tau = diag(1./(2*obj.kernelBandwidth.^2));
            if strcmp(obj.kernelType, 'squaredExponential')
                k = exp(- kernelDiff*tau*kernelDiff');
            else
                error('Unknown kernel type')
            end
        end

        
        
        %% plot functions
        function [p] = plotTrainingInput(obj, samples, titl)
            %Load microstructures
            samplesTemp = min(samples):max(samples);
            cond = obj.trainingDataMatfile.cond(:, samplesTemp);
            samples = samples - min(samples) + 1;
            f = figure;
            obj = obj.loadTrainingData;
            xLines = cumsum(obj.coarseGridVectorX)*obj.nElFX;
            yLines = cumsum(obj.coarseGridVectorY)*obj.nElFY;
            for i = 1:16
                subplot(4,4,i);
                p(i) = imagesc(reshape(cond(:, samples(i)), obj.fineScaleDomain.nElX, obj.fineScaleDomain.nElY));
                grid off;
                axis square;
                xticks({});
                yticks({});
                if nargin > 2
                    %to plot numerical title
                    title(num2str(titl(i)));
                end
                for x = 1:(numel(obj.coarseGridVectorX) - 1)
                    line([xLines(x), xLines(x)], [0, obj.nElFX], 'Color', 'w')
                end
                for y = 1:(numel(obj.coarseGridVectorY) - 1)
                    line([0, obj.nElFX], [yLines(y), yLines(y)], 'Color', 'w')
                end
            end
            
        end

        function [p, im] = plotTrainingOutput(obj, samples, titl)
            %Load microstructures
            samplesTemp = min(samples):max(samples);
            Tf = obj.trainingDataMatfile.Tf(:, samplesTemp);
            cond = obj.trainingDataMatfile.cond(:, samplesTemp);
            samples = samples - min(samples) + 1;
            min_Tf = min(min(Tf(:, samples)));
            max_Tf = max(max(Tf(:, samples)));
            %to use same color scale
            cond = ((min_Tf - max_Tf)/(min(min(cond)) - max(max(cond))))*cond + max_Tf - ...
                ((min_Tf - max_Tf)/(min(min(cond)) - max(max(cond))))*max(max(cond));
            f = figure;
            obj = obj.loadTrainingData;
            xLines = cumsum(obj.coarseGridVectorX)*obj.nElFX;
            yLines = cumsum(obj.coarseGridVectorY)*obj.nElFY;
            for i = 1:6
                subplot(2,3,i);
                p(i) = surf(reshape(Tf(:, samples(i)), (obj.fineScaleDomain.nElX + 1),...
                    (obj.fineScaleDomain.nElY + 1)));
                caxis([min_Tf, max_Tf])
                hold
                im(i) = imagesc(reshape(cond(:, samples(i)), obj.fineScaleDomain.nElX, obj.fineScaleDomain.nElY));
                p(i).LineStyle = 'none';
                grid on;
                axis tight;
                box on;
                axis square;
                zlim([min_Tf, max_Tf])
%                 xlabel('x')
%                 ylabel('y')
                zlabel('u')
                xticklabels({})
                yticklabels({})
%                 xticks({});
%                 yticks({});
                if nargin > 2
                    %to plot numerical title
                    title(num2str(titl(i)));
                end
%                 for x = 1:(numel(obj.coarseGridVectorX) - 1)
%                     line([xLines(x), xLines(x)], [0, obj.nElFX], 'Color', 'w')
%                 end
%                 for y = 1:(numel(obj.coarseGridVectorY) - 1)
%                     line([0, obj.nElFX], [yLines(y), yLines(y)], 'Color', 'w')
%                 end
            end
            
            for i = 1:10
                f = figure;
                subplot(1,2,1)
                p(i) = imagesc(reshape(cond(:, samples(i)), obj.fineScaleDomain.nElX, obj.fineScaleDomain.nElY));
                grid off;
                axis square;
                xticks({});
                yticks({});
                subplot(1,2,2)
                q(i) = surf(reshape(Tf(:, samples(i)), (obj.fineScaleDomain.nElX + 1),...
                    (obj.fineScaleDomain.nElY + 1)));
                q(i).LineStyle = 'none';
                grid on;
                box on;
                axis square;
                axis tight;
                xticks([64 128 192]);
                yticks([64 128 192]);
                zticks(100:100:800);
                xticklabels({});
                yticklabels({});
                zticklabels({});
                zlim([0 800])
                print(f, strcat('~/images/uncecomp17/fineScaleSample', num2str(i)), '-dpng', '-r300')
            end
            
        end

        function p = plot_p_c_regression(obj)
            %Plots regressions of single features to the data <X>_q
            totalFeatures = size(obj.featureFunctions, 2) + size(obj.globalFeatureFunctions, 2) + ...
                obj.latentDim;
%             %Setting up handles to feature function vectors
%             function phi_vec = phiVec(xk, k)
%                 phi_vec = zeros(totalFeatures, 1);
%                 for feat_loc = 1:size(obj.featureFunctions, 2)
%                     phi_vec(feat_loc) = obj.featureFunctions{k, feat_loc}(xk);
%                 end
%             end
            for feature = 1:min([4, totalFeatures])
                f = figure;
                if obj.useKernels
                    if size(obj.globalFeatureFunctions, 2)
                        error('Not yet generalized for global features')
                    end
                    load(strcat('./persistentData/', 'train', 'DesignMatrix.mat'), 'designMatrix');
                    %Setting up handles to kernel functions
                    for k = 1:obj.coarseScaleDomain.nEl
                        for n = 1:obj.nTrain
                            %phi_vec must be a row vector!!
                            kernelDiff{n, k} = @(phi_vec) phi_vec - designMatrix{n}(k, :);
                            kernelHandle{n, k} = @(phi_vec) obj.kernelFunction(kernelDiff{n, k}(phi_vec));
                        end
                    end
                end
                k = 1;
                if strcmp(obj.mode, 'useLocal')
                    mink = Inf*ones(obj.coarseScaleDomain.nEl, 1);
                    maxk = -Inf*ones(obj.coarseScaleDomain.nEl, 1);
                    for i = 1:obj.coarseScaleDomain.nElX
                        for j = 1:obj.coarseScaleDomain.nElY
                            subplot(obj.coarseScaleDomain.nElX, obj.coarseScaleDomain.nElY, k);
                            for s = 1:obj.nTrain
                                yData(k, s) = obj.XMean(k, s);
                                for l = 1:totalFeatures
                                    if l~= feature
                                        yData(k, s) = yData(k, s) -...
                                            obj.theta_c.theta(totalFeatures*(k - 1) + l)*...
                                            obj.designMatrix{s}(k, totalFeatures*(k - 1) + l);
                                    end
                                end
                                plot(obj.designMatrix{s}(k, totalFeatures*(k - 1) + feature), yData(k, s), 'xb')
                                if(obj.designMatrix{s}(k, totalFeatures*(k - 1) + feature) < mink(k))
                                    mink(k) = obj.designMatrix{s}(k, totalFeatures*(k - 1) + feature);
                                elseif(obj.designMatrix{s}(k, totalFeatures*(k - 1) + feature) > maxk(k))
                                    maxk(k) = obj.designMatrix{s}(k, totalFeatures*(k - 1) + feature);
                                end
                                hold on;
                            end
                            x = linspace(mink(k), maxk(k), 100);
                            y = 0*x;
                            
                            if obj.useKernels
                                if(size(obj.featureFunctions, 2) == 1)
                                    for m = 1:length(x)
                                        for n = 1:obj.nTrain
                                            y(m) = y(m) + kernelHandle{n, k}(x(m))*...
                                                obj.theta_c.theta((n - 1)*obj.coarseScaleDomain.nEl + k);
                                        end
                                    end
                                    plot(x, y);
                                    axis tight;
                                    axis square;
                                    xl = xlabel('Feature function output $\phi_i$');
                                    xl.Interpreter = 'latex';
                                    yl = ylabel('$<X_k> - \sum_{j\neq i} \theta_j \phi_j$');
                                    yl.Interpreter = 'latex';
                                    k = k + 1;
                                else
                                    error('Not yet generalized for more than 1 feature')
                                end
                            else
                                useOffset = false;
                                if useOffset
                                    %it is important that the offset feature phi(lambda) = 1 is the very
                                    %first feature
                                    y = obj.theta_c.theta(totalFeatures*(k - 1) + 1) +...
                                        obj.theta_c.theta(totalFeatures*(k - 1) + feature)*x;
                                else
                                    y = obj.theta_c.theta(totalFeatures*(k - 1) + feature)*x;
                                end
                                plot(x, y);
                                axis tight;
                                axis square;
                                xl = xlabel('Feature function output $\phi_i$');
                                xl.Interpreter = 'latex';
                                yl = ylabel('$<X_k> - \sum_{j\neq i} \theta_j \phi_j$');
                                yl.Interpreter = 'latex';
                                k = k + 1;
                            end
                        end
                    end
                elseif strcmp(obj.mode, 'none')
                    for i = 1:obj.coarseScaleDomain.nElX
                        for j = 1:obj.coarseScaleDomain.nElY
                            for s = 1:obj.nTrain
                                if obj.useKernels
                                    if(totalFeatures == 1)
                                        plot(designMatrix{s}(k, feature), obj.XMean(k, s), 'xb')
                                    else
                                        if(feature == 1)
                                            plot3(designMatrix{s}(k, 1), designMatrix{s}(k, 2), obj.XMean(k, s),...
                                                'xb', 'linewidth', 2)
                                        end
                                    end
                                else
                                    plot(obj.designMatrix{s}(k, feature), obj.XMean(k, s), 'xb')
                                end
                                hold on;
                            end
                            k = k + 1;
                        end
                    end
                    if obj.useKernels
                        if(totalFeatures == 1)
                            x = linspace(min(min(cell2mat(designMatrix))),...
                                max(max(cell2mat(designMatrix))), 100);
                            y = 0*x;
                        else
                            designMatrixArray = cell2mat(designMatrix');
                            %Maxima and minima of first and second feature
                            max1 = max(max(designMatrixArray(:, 1:totalFeatures:end)));
                            max2 = max(max(designMatrixArray(:, 2:totalFeatures:end)));
                            min1 = min(min(designMatrixArray(:, 1:totalFeatures:end)));
                            min2 = min(min(designMatrixArray(:, 2:totalFeatures:end)));
                            [X, Y] = meshgrid(linspace(min1, max1, 30), linspace(min2, max2, 30));
                        end
                    else
                        x = linspace(min(min(cell2mat(obj.designMatrix))),...
                            max(max(cell2mat(obj.designMatrix))), 100);
                        y = 0*x;
                    end
                    if obj.useKernels
                        if(totalFeatures == 1)
                            for m = 1:length(x)
                                for n = 1:obj.nTrain
                                    for h = 1:obj.coarseScaleDomain.nEl
                                        y(m) = y(m) + kernelHandle{n, h}(x(m))*...
                                            obj.theta_c.theta((n - 1)*obj.coarseScaleDomain.nEl + h);
                                    end
                                end
                            end
                            plot(x, y);
                            axis tight;
                        else
                            if(feature == 1)
                                %Plot first two features as surface
                                Z = 0*X;  %prealloc
                                for c = 1:size(Z, 2)
                                    for r = 1:size(Z, 1)
                                        %Projection onto the first two features
                                        phi_vec = zeros(1, size(designMatrix{1}, 2));
                                        phi_vec(1:2) = [X(r, c) Y(r, c)];
                                        for n = 1:obj.nTrain
                                            for h = 1:obj.coarseScaleDomain.nEl
                                                Z(r, c) = Z(r, c) + kernelHandle{n, h}(phi_vec)*...
                                                    obj.theta_c.theta((n - 1)*obj.coarseScaleDomain.nEl + h);
                                            end
                                        end
                                    end
                                end
                                surf(X, Y, Z)
                                axis square;
                                axis tight;
                                box on;
                            end
                        end
                    else
                        y = obj.theta_c.theta(feature)*x;
                        plot(x, y);
                        axis tight;
                    end
                end
            end
        end
        
        function obj = plotTheta(obj, figHandle)
            %Plots the current theta_c
            if isempty(obj.thetaArray)
                obj.thetaArray = obj.theta_c.theta';
            else
                if(size(obj.theta_c.theta, 1) > size(obj.thetaArray, 2))
                    %New basis function included. Expand array
                    obj.thetaArray = [obj.thetaArray, zeros(size(obj.thetaArray, 1),...
                        numel(obj.theta_c.theta) - size(obj.thetaArray, 2))];
                    obj.thetaArray = [obj.thetaArray; obj.theta_c.theta'];
                else
                    obj.thetaArray = [obj.thetaArray; obj.theta_c.theta'];
                end
            end
            if isempty(obj.thetaHyperparamArray)
                obj.thetaHyperparamArray = obj.thetaPriorHyperparam';
            else
                if(size(obj.thetaPriorHyperparam, 1) > size(obj.thetaHyperparamArray, 2))
                    %New basis function included. Expand array
                    obj.thetaHyperparamArray = [obj.thetaHyperparamArray, zeros(size(obj.thetaHyperparamArray, 1),...
                        numel(obj.thetaPriorHyperparam) - size(obj.thetaHyperparamArray, 2))];
                    obj.thetaHyperparamArray = [obj.thetaHyperparamArray; obj.thetaPriorHyperparam'];
                else
                    obj.thetaHyperparamArray = [obj.thetaHyperparamArray; obj.thetaPriorHyperparam'];
                end
            end
            if isempty(obj.sigmaArray)
                obj.sigmaArray = diag(obj.theta_c.Sigma)';
            else
                obj.sigmaArray = [obj.sigmaArray; diag(obj.theta_c.Sigma)'];
            end
%            figure(figHandle);
            sb = subplot(3, 2, 1, 'Parent', figHandle);
            plot(obj.thetaArray, 'linewidth', 1, 'Parent', sb)
            axis tight;
            sb = subplot(3,2,2, 'Parent', figHandle);
            plot(obj.theta_c.theta, 'linewidth', 1, 'Parent', sb)
            axis tight;
            sb = subplot(3,2,3, 'Parent', figHandle);
            semilogy(sqrt(obj.sigmaArray), 'linewidth', 1, 'Parent', sb)
            axis tight;
            sb = subplot(3,2,4, 'Parent', figHandle);
            imagesc(reshape(diag(sqrt(obj.theta_c.Sigma)), obj.coarseScaleDomain.nElX,...
                obj.coarseScaleDomain.nElY), 'Parent', sb)
            title('\sigma_k^2')
            colorbar
            grid off;
            axis tight;
            sb = subplot(3,2,5, 'Parent', figHandle);
            plot(obj.thetaHyperparamArray, 'linewidth', 1, 'Parent', sb)
            axis tight;
            sb = subplot(3,2,6, 'Parent', figHandle);
            plot(obj.thetaPriorHyperparam, 'linewidth', 1, 'Parent', sb)
            axis tight;
            drawnow
        end

        function obj = addLinearFilterFeature(obj)            
            assert(strcmp(obj.mode, 'useLocal'),...
                'Error: sequential addition of linear filters only working in useLocal mode');
            
%             sigma2Inv_vec = (1./diag(obj.theta_c.Sigma));
            XMeanMinusPhiThetac = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
            for i = 1:obj.nTrain
                XMeanMinusPhiThetac(:, i) = obj.XMean(:, i) - obj.designMatrix{i}*obj.theta_c.theta;
            end
            
            %We use different linear filters for different macro-cells k
            w{1} = 0;
            w = repmat(w, obj.coarseScaleDomain.nEl, 1);
            E = zeros(1, obj.coarseScaleDomain.nEl);
            for m = 1:obj.coarseScaleDomain.nEl
                for i = 1:obj.nTrain
%                     w{m} = w{m} + sigma2Inv_vec(m)*XMeanMinusPhiThetac(m, i)*obj.xk{i, m}(:);
                    %should be still correct without the sigma and subsequent normalization
                    w{m} = w{m} + XMeanMinusPhiThetac(m, i)*obj.xk{i, m}(:);
                end
                %normalize
                E(m) = norm(w{m});
%                 w{m} = w{m}'/E(m);
                w{m} = w{m}'/norm(w{m}, 1);
            end
            
            %save w
            filename = './data/w.mat';
            if exist(filename, 'file')
                load(filename)  %load w_all
            else
                w_all = {};
            end
            %append current w's as cell array column index
            nFeaturesBefore = size(obj.featureFunctions, 2);
            nLinFiltersBefore = size(w_all, 2);
            for m = 1:obj.coarseScaleDomain.nEl
                w_all{m, nLinFiltersBefore + 1} = w{m};
                obj.featureFunctions{m, nFeaturesBefore + 1} = @(lambda) sum(w{m}'.*...
                    conductivityTransform(lambda(:), obj.conductivityTransformation));
            end
            
            save(filename, 'w_all');
            %save E
            filename = './data/E';
            save(filename, 'E', '-ascii', '-append');
            
            f = figure;
            for m = 1:obj.coarseScaleDomain.nEl
                subplot(obj.coarseScaleDomain.nElX, obj.coarseScaleDomain.nElY, m);
                imagesc(reshape(w{m}, size(obj.xk{1, m})))
                axis square
                grid off
                xticks({})
                yticks({})
                colorbar
            end
            drawnow
            
            %% recompute design matrices
            %this can be done more efficiently!
            obj = obj.computeDesignMatrix('train', true);
            
            %% append theta-value
            nTotalFeaturesAfter = size(obj.designMatrix{1}, 2);
            theta_new = zeros(nTotalFeaturesAfter, 1);
            j = 1;
            for i = 1:nTotalFeaturesAfter
                if(mod(i, nTotalFeaturesAfter/obj.coarseScaleDomain.nEl) == 0)
                    theta_new(i) = 0;
                else
                    theta_new(i) = obj.theta_c.theta(j);
                    j = j + 1;
                end
            end
            obj.theta_c.theta = theta_new;
        end

        function obj = addGlobalLinearFilterFeature(obj)
            assert(strcmp(obj.mode, 'useLocal'),...
                'Error: sequential addition of linear filters only working in useLocal mode');
            XMeanMinusPhiThetac = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
            for i = 1:obj.nTrain
                XMeanMinusPhiThetac(:, i) = obj.XMean(:, i) - obj.designMatrix{i}*obj.theta_c.theta;
            end
            
            conductivity = obj.trainingDataMatfile.cond(:, obj.trainingSamples);
            
            %We use different linear filters for different macro-cells k
            w{1} = 0;
            w = repmat(w, obj.coarseScaleDomain.nEl, 1);
            EGlobal = zeros(1, obj.coarseScaleDomain.nEl);
            for m = 1:obj.coarseScaleDomain.nEl
                for i = 1:obj.nTrain
                    w{m} = w{m} + XMeanMinusPhiThetac(m, i)*conductivityTransform(conductivity(:, i),...
                        obj.conductivityTransformation);
                end
                %normalize
                EGlobal(m) = norm(w{m});
%                 w{m} = w{m}'/EGlobal(m);
                w{m} = w{m}'/norm(w{m}, 1);
            end
            
            %save w
            filename = './data/wGlobal.mat';
            if exist(filename, 'file')
                load(filename)  %load w_allGlobal
            else
                w_allGlobal = {};
            end
            %append current w's as cell array column index
            nGlobalFeaturesBefore = size(obj.globalFeatureFunctions, 2);
            nGlobalLinFiltersBefore = size(w_allGlobal, 2);
            for m = 1:obj.coarseScaleDomain.nEl
                w_allGlobal{m, nGlobalLinFiltersBefore + 1} = w{m};
                obj.globalFeatureFunctions{m, nGlobalFeaturesBefore + 1} = @(lambda) sum(w{m}'.*...
                    conductivityTransform(lambda(:), obj.conductivityTransformation));
            end
            
            save(filename, 'w_allGlobal');
            %save E
            filename = './data/EGlobal';
            save(filename, 'EGlobal', '-ascii', '-append');
            
            f = figure;
            for m = 1:obj.coarseScaleDomain.nEl
                subplot(obj.coarseScaleDomain.nElX, obj.coarseScaleDomain.nElY, m);
                imagesc(reshape(w{m}, obj.fineScaleDomain.nElX, obj.fineScaleDomain.nElY))
                axis square
                grid off
                xticks({})
                yticks({})
                colorbar
            end
            drawnow
            
            %% recompute design matrices
            %this can be done more efficiently!
            obj = obj.computeDesignMatrix('train', true);
            
            %% extend theta vector
            nTotalFeaturesAfter = size(obj.designMatrix{1}, 2);
            theta_new = zeros(nTotalFeaturesAfter, 1);
            j = 1;
            for i = 1:nTotalFeaturesAfter
                if(mod(i, nTotalFeaturesAfter/obj.coarseScaleDomain.nEl) == 0)
                    theta_new(i) = 0;
                else
                    theta_new(i) = obj.theta_c.theta(j);
                    j = j + 1;
                end
            end
            obj.theta_c.theta = theta_new;
            
        end

        function [X, Y, corrX_log_p_cf, corrX, corrY, corrXp_cf, corrpcfX, corrpcfY] = findMeshRefinement(obj)
            %Script to sample d_log_p_cf under p_c to find where to refine mesh next

            Tf = obj.trainingDataMatfile.Tf(:, obj.nStart:(obj.nStart + obj.nTrain - 1));
            
            obj = obj.loadTrainedParams;
            theta_cfTemp = obj.theta_cf;
            
            %comment this for inclusion of variances S of p_cf
            theta_cfTemp.S = ones(size(theta_cfTemp.S));
            
            theta_cfTemp.Sinv = sparse(1:obj.fineScaleDomain.nNodes, 1:obj.fineScaleDomain.nNodes, 1./theta_cfTemp.S);
            theta_cfTemp.Sinv_vec = 1./theta_cfTemp.S;
            %precomputation to save resources
            theta_cfTemp.WTSinv = theta_cfTemp.W'*theta_cfTemp.Sinv;
            theta_cfTemp.sumLogS = sum(log(theta_cfTemp.S));
            
            %% Compute design matrices
            if isempty(obj.designMatrix)
                obj = obj.computeDesignMatrix('train');
            end
            
            nSamples = 1000;
            d_log_p_cf_mean = 0;
            log_p_cf_mean = 0;
            p_cfMean = 0;
            p_cfSqMean = 0;
            log_p_cfSqMean = 0;
            d_log_p_cf_sqMean = 0;
            k = 1;
            XsampleMean = 0;
            XsampleSqMean = 0;
            Xlog_p_cf_mean = 0;
            Xp_cfMean = 0;
            for i = obj.nStart:(obj.nStart + obj.nTrain - 1)
                mu_i = obj.designMatrix{i}*obj.theta_c.theta;
                XsampleMean = ((i - 1)/i)*XsampleMean + (1/i)*mu_i;
                XsampleSqMean = ((i - 1)/i)*XsampleSqMean + (1/i)*mu_i.^2;
                for j = 1:nSamples
                    Xsample = mvnrnd(mu_i, obj.theta_c.Sigma)';
                    conductivity = conductivityBackTransform(Xsample, obj.conductivityTransformation);
                    [lg_p_cf, d_log_p_cf] = log_p_cf(Tf(:, i), obj.coarseScaleDomain, conductivity,...
                        theta_cfTemp, obj.conductivityTransformation);
                    d_log_p_cf_mean = ((k - 1)/k)*d_log_p_cf_mean + (1/k)*d_log_p_cf;
                    d_log_p_cf_sqMean = ((k - 1)/k)*d_log_p_cf_sqMean + (1/k)*d_log_p_cf.^2;
                    log_p_cf_mean = ((k - 1)/k)*log_p_cf_mean + (1/k)*lg_p_cf;
                    log_p_cfSqMean = ((k - 1)/k)*log_p_cfSqMean + (1/k)*lg_p_cf^2;
                    p_cfMean = ((k - 1)/k)*p_cfMean + (1/k)*exp(lg_p_cf);
                    p_cfSqMean = ((k - 1)/k)*p_cfSqMean + (1/k)*exp(2*lg_p_cf);
                    Xlog_p_cf_mean = ((k - 1)/k)*Xlog_p_cf_mean + (1/k)*Xsample*lg_p_cf;
                    Xp_cfMean = ((k - 1)/k)*Xp_cfMean + (1/k)*Xsample*exp(lg_p_cf);
                    k = k + 1;
                end
            end
            covX_log_p_cf = Xlog_p_cf_mean - XsampleMean*log_p_cf_mean;
            var_log_p_cf = log_p_cfSqMean - log_p_cf_mean^2;
            varX = XsampleSqMean - XsampleMean.^2;
            corrX_log_p_cf = covX_log_p_cf./(sqrt(var_log_p_cf)*sqrt(varX));
            
            covXp_cf = Xp_cfMean - XsampleMean*p_cfMean
            var_p_cf = p_cfSqMean - p_cfMean^2
            corrXp_cf = covXp_cf./(sqrt(var_p_cf)*sqrt(varX))
            
            d_log_p_cf_mean
            d_log_p_cf_var = d_log_p_cf_sqMean - d_log_p_cf_mean.^2
            d_log_p_cf_std = sqrt(d_log_p_cf_var)
            d_log_p_cf_err = d_log_p_cf_std/sqrt(nSamples*obj.nTrain)
            d_log_p_cf_sqMean
            load('./data/noPriorSigma')
            noPriorSigma
            log_noPriorSigma = log(noPriorSigma)
            
            disp('Sum of grad squares in x-direction:')
            for i = 1:obj.coarseScaleDomain.nElY
                X(i) = sum(d_log_p_cf_sqMean(((i - 1)*obj.coarseScaleDomain.nElX + 1):(i*obj.coarseScaleDomain.nElX)));
                corrX(i) = sum(abs(corrX_log_p_cf(((i - 1)*obj.coarseScaleDomain.nElX + 1):...
                    (i*obj.coarseScaleDomain.nElX))));
                corrpcfX(i) = sum((corrXp_cf(((i - 1)*obj.coarseScaleDomain.nElX + 1):...
                    (i*obj.coarseScaleDomain.nElX))).^2);
            end
            
            disp('Sum of grad squares in y-direction:')
            for i = 1:obj.coarseScaleDomain.nElX
                Y(i) = sum(d_log_p_cf_sqMean(i:obj.coarseScaleDomain.nElX:...
                    ((obj.coarseScaleDomain.nElY - 1)*obj.coarseScaleDomain.nElX + i)));
                corrY(i) = sum(abs(corrX_log_p_cf(i:obj.coarseScaleDomain.nElX:...
                    ((obj.coarseScaleDomain.nElY - 1)*obj.coarseScaleDomain.nElX + i))));
                corrpcfY(i) = sum((corrXp_cf(i:obj.coarseScaleDomain.nElX:...
                    ((obj.coarseScaleDomain.nElY - 1)*obj.coarseScaleDomain.nElX + i))).^2);
            end
        end

        
        
        %% Setter functions
        function obj = setConductivityDistributionParams(obj, condDistParams)
            obj.conductivityDistributionParams = condDistParams;
            obj = obj.generateFineScaleDataPath;
        end

        function obj = setBoundaryConditions(obj, boundaryConditions)
            %Coefficients of boundary condition functions must be given as string
            assert(ischar(boundaryConditions), 'boundaryConditions must be given as string');
            obj.boundaryConditions = boundaryConditions;
            obj = obj.genBoundaryConditionFunctions;
        end

        function obj = setFeatureFunctions(obj)
            %Set up feature function handles;
            %First cell array index is for macro-cell. This allows different features for different
            %macro-cells

            addpath('./featureFunctions')   %Path to feature function library
            conductivities = [obj.lowerConductivity obj.upperConductivity];
            log_cutoff = 1e-5;
            obj.featureFunctions = {};
            obj.globalFeatureFunctions = {};
            %constant bias
            for k = 1:obj.coarseScaleDomain.nEl
                nFeatures = 0;
%                 obj.featureFunctions{k, nFeatures + 1} = @(lambda) 1;
%                 nFeatures = nFeatures + 1;
%                 
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    SCA(lambda, conductivities, obj.conductivityTransformation);
                nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     maxwellGarnett(lambda, conductivities, obj.conductivityTransformation, 'lo');
% %                 nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     differentialEffectiveMedium(lambda, conductivities, obj.conductivityTransformation, 'lo');
% %                 nFeatures = nFeatures + 1;
%                 
%                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
%                     linealPath(lambda, 3, 'x', 2, conductivities);
%                 nFeatures = nFeatures + 1;
%                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
%                     linealPath(lambda, 3, 'y', 2, conductivities);
%                 nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     linealPath(lambda, 3, 'x', 1, conductivities);
% %                 nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     linealPath(lambda, 3, 'y', 1, conductivities);
% %                 nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     linealPath(lambda, 6, 'x', 1, conductivities);
% %                 nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     linealPath(lambda, 6, 'y', 1, conductivities);
% %                 nFeatures = nFeatures + 1;
% % 
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     numberOfObjects(lambda, conductivities, 'hi');
% %                 nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     nPixelCross(lambda, 'y', 1, conductivities, 'mean');
% % 				nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     nPixelCross(lambda, 'x', 1, conductivities, 'mean');
% % 				nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     maxExtent(lambda, conductivities, 'hi', 'y');
% %                 nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     maxExtent(lambda, conductivities, 'hi', 'x');
% %                 nFeatures = nFeatures + 1;
% %                 
% % 
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     conductivityTransform(generalizedMean(lambda, -1), obj.conductivityTransformation);
% %                 nFeatures = nFeatures + 1;
% % 
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda) log(meanImageProps(lambda,...
% %                     conductivities, 'hi', 'ConvexArea', 'max') + log_cutoff);
% %                 nFeatures = nFeatures + 1;
%                 
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda) ...
% %                     connectedPathExist(lambda, 2, conductivities, 'x', 'invdist');
% %                 nFeatures = nFeatures + 1;
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda) ...
% %                     connectedPathExist(lambda, 2, conductivities, 'y', 'invdist');
% %                 nFeatures = nFeatures + 1;
%                 
% %                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
% %                     log(specificSurface(lambda, 2, conductivities, [obj.nElFX obj.nElFY]) + log_cutoff);
% %                 nFeatures = nFeatures + 1;
% 
%                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
%                     gaussLinFilt(lambda);
%                 nFeatures = nFeatures + 1;

%                 obj.featureFunctions{k, nFeatures + 1} = @(lambda) std(lambda(:));
%                 nFeatures = nFeatures + 1;
            end
            
            obj.secondOrderTerms = zeros(nFeatures, 'logical');
%             obj.secondOrderTerms(2, 2) = true;
%             obj.secondOrderTerms(2, 3) = true;
%             obj.secondOrderTerms(3, 3) = true;
            assert(sum(sum(tril(obj.secondOrderTerms, -1))) == 0, 'Second order matrix must be upper triangular')
            
        end

        function obj = setCoarseGrid(obj, coarseGridX, coarseGridY)
            %coarseGridX and coarseGridY are coarse model grid vectors
            obj.coarseGridVectorX = coarseGridX;
            obj.coarseGridVectorY = coarseGridY;
        end
    end
    
end


















