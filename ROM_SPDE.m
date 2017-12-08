classdef ROM_SPDE
    %Class for ROM SPDE model
    
    properties  %public
        %% Finescale data specifications
        %Finescale system size
        nElFX = 256;
        nElFY = 256;
        %Finescale conductivities, binary material
        lowerConductivity = 1;
        upperConductivity = 2;
        %Conductivity field distribution type
        conductivityDistribution = 'squaredExponential';
        %Boundary condition functions; evaluate those on boundaries to get boundary conditions
        boundaryTemperature;
        boundaryHeatFlux;
        bcArrayTrain;   %boundary condition cell array for training data
        bcArrayTest;    %boundary condition cell array for test data
        
        naturalNodes;
        %Directory where finescale data is stored; specify basename here
        fineScaleDataPath = '~/cluster/matlab/data/fineData/';
        %matfile handle
        trainingDataMatfile;
        testDataMatfile;
        %Finescale Domain object
        fineScaleDomain;
        %Array holding fine scale data output; possibly large
        fineScaleDataOutput;
        %number of samples per generated matfile
        nSets = [4096 4096];
        %Output data characteristics
        outputVariance;
        outputMean;
        meanOutputVariance;
        E;                  %Mapping from fine to coarse cell index
        neighborDictionary; %Gives neighbors of macrocells
        %% Model training parameters
        nStart = 1;             %first training data sample in file
        nTrain = 16;            %number of samples used for training
        mode = 'none';          %useNeighbor, useLocalNeighbor, useDiagNeighbor, useLocalDiagNeighbor, useLocal, global
                                %global: take whole microstructure as feature function input, not
                                %only local window (only recommended for pooling)
        inferenceMethod = 'variationalInference';        %E-step inference method. variationalInference or monteCarlo
        
        %% Sequential addition of linear filters
        linFilt
        
        useAutoEnc = false;     %Use autoencoder information? Do not forget to pre-train autoencoder!
        globalPcaComponents = 3;   %Principal components of the whole microstructure used as features 
        localPcaComponents = 7;     %Principal components of single macro-cell used as features
        pcaSamples = 4096;
        secondOrderTerms;
        mix_S = 0;              %To slow down convergence of S
        mix_theta = 0;
        mix_sigma = 0;
        fixSigmaInit = 0;       %Fix sigma in initial iterations
        
        %% Prior specifications
        sigmaPriorType = 'none';    %none, delta, expSigSq
        sigmaPriorHyperparam = 1;
        thetaPriorType = 'RVM';
        thetaPriorHyperparam = [0 1e-30];
        
        %% Model parameters
        theta_c;
        theta_cf;
        free_W = false;
        featureFunctions;       %Cell array containing local feature function handles
        globalFeatureFunctions; %cell array with handles to global feature functions
        convectionFeatureFunctions;       %Cell array containing local convection feature function handles
        globalConvectionFeatureFunctions; %cell array with handles to global convection feature functions
        kernelHandles
        %transformation of finescale conductivity to real axis
        conductivityTransformation;
        latentDim = 0;              %If autoencoder is used
        sumPhiTPhi;             %Design matrix precomputation
        padding = 0;           %How many pixels around macro-cell should be considered in local features?
        
        %% Feature function rescaling parameters
        featureScaling = 'normalize'; %'standardize' for zero mean and unit variance of features, 'rescale' to have
                                        %all between 0 and 1, 'normalize' to have unit variance only
                                        %(and same mean as before)
        featureFunctionMean;
        featureFunctionSqMean;
        featureFunctionMin;
        featureFunctionMax;
        loadDesignMatrix = false;
        useKernels = false;              %Use linear combination of kernels in feature function space
        kernelBandwidth = 2;              %only used if no rule of thumb is taken
        bandwidthSelection = 'silverman'  %'fixed' for fixed kernel bandwidth, 'silverman' for
                                          %silverman's rule of thumb 'scott' for scott's rule of thumb,
                                          %see wikipedia on multivariate kernel density estimation
        kernelType = 'squaredExponential';
        
        %% Prediction parameters
        nSamples_p_c = 1000;
        useLaplaceApproximation = true;   %Use Laplace approx around MAP for prediction?
        testSamples = [1:1024];       %pick out specific test samples here
        trainingSamples;   %pick out specific training samples here
        
        %% Prediction outputs
        predMeanArray;
        predVarArray;
        meanPredMeanOutput;                %mean predicted mean of output field
        meanSquaredDistance;               %mean squared distance of predicted mean to true solution
        meanSquaredDistanceField;
        meanSquaredDistanceError;          %Monte Carlo error
        meanLogLikelihood;
        meanLogPerplexity;
        meanPerplexity;
        meanMahalanobisError;
        meanEffCond;
        
        %% Finescale data- only load this to memory when needed!
        lambdak
        xk
        
        %% Computational quantities
        varExpect_p_cf_exp_mean;
        XMean;                  %Expected value of X under q
        XSqMean;                %<X^2>_q
        mean_TfTcT;
        mean_TcTcT;
        thetaArray;
        thetaHyperparamArray;
        sigmaArray;
        
        EM_iterations = 1;
        epoch = 0;      %how often every data point has been seen
        epoch_old = 0;  %To check if epoch has changed
        maxEpochs;      %Maximum number of epochs
        
        useConvection = false;      %Include a convection term to the pde? Uses convection term in coarse model in
                                   %training/prediction
        useConvectionData = false;
    end
    
    
    properties(SetAccess = private)
        %% finescale data specifications
        conductivityLengthScaleDist = 'delta';      %delta for fixed length scale, lognormal for rand
        conductivityDistributionParams = {-1 [.01 .01] 1};
        advectionDistributionParams = [10, 15];   %mu and sigma for advection field coefficients
        %{volumeFraction, correlationLength, sigma_f2}
        %for log normal length scale, the
        %length scale parameters are log normal mu and
        %sigma
        %Coefficients giving boundary conditions, specify as string
        boundaryConditions = '[0 800 1200 -2000]';
        boundaryConditionVariance = [0 0 0 0];
        
        %% Coarse model specifications
        coarseScaleDomain;
        coarseGridVectorX = (1/4)*ones(1, 4);
        coarseGridVectorY = (1/4)*ones(1, 4);
        
        %Design matrices. Cell index gives data point, row index coarse cell, and column index
        %feature function
        designMatrix
        originalDesignMatrix    %design matrix without any locality mode
        testDesignMatrix    %design matrices on independent test set
        kernelMatrix
    end
    
    
    methods
        function obj = ROM_SPDE(mode)
            %Constructor
            %Create data directory
            if ~exist('./data/', 'dir')
                mkdir('./data/');
            end
            
            %add needed paths
            addpath('./aux');
            
            %set up path
            obj = obj.genFineScaleDataPath;
            
            %Set handles to boundary condition functions
            obj = obj.genBoundaryConditionFunctions;
            obj.naturalNodes = [2:(2*obj.nElFX + 2*obj.nElFY)];
            
            %Set up coarseScaleDomain; must be done after boundary conditions are set up
            obj = obj.genCoarseDomain;
            
            %prealloc
            if obj.useConvection
                obj.XMean = zeros(3*obj.coarseScaleDomain.nEl, obj.nTrain);
                obj.XSqMean = ones(3*obj.coarseScaleDomain.nEl, obj.nTrain);
            else
                obj.XMean = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
                obj.XSqMean = ones(obj.coarseScaleDomain.nEl, obj.nTrain);
            end
            %Set up default value for test samples
            obj.testSamples = 1:obj.nSets(2);
            obj.nStart = randi(obj.nSets(1) - obj.nTrain, 1)
            obj.trainingSamples = obj.nStart:(obj.nStart + obj.nTrain - 1);
            
            %Set conductivity transformation
            obj.conductivityTransformation.anisotropy = false;
            obj.conductivityTransformation.type = 'logit';
            if strcmp(obj.conductivityTransformation.type, 'log')
                obj.conductivityTransformation.limits = [1e-6 1e6];
            elseif strcmp(obj.conductivityTransformation.type, 'logit')
                obj.conductivityTransformation.limits =...
                    [(1 - 1e-2)*obj.lowerConductivity (1 + 1e-2)*obj.upperConductivity];
            else
                obj.conductivityTransformation.limits = [1e-8 1e8];
            end
            conductivityTransformation = obj.conductivityTransformation;
            save('./data/conductivityTransformation', 'conductivityTransformation');
            
            obj.linFilt.totalUpdates = 0;
            if ~strcmp(mode, 'genData')
                %Load fine scale domain and set boundary conditions
                obj = obj.loadTrainingData;
                %Set up feature function handles
                %obj = obj.setFeatureFunctions;
                %Prealloc
                obj.varExpect_p_cf_exp_mean = zeros(obj.fineScaleDomain.nNodes, 1);
            end
            
            %Check
            if(strcmp(obj.thetaPriorType, 'sharedRVM') && ~strcmp(obj.mode, 'useLocal'))
                error('sharedRVM prior only allowed for useLocal mode')
            end
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
            obj.fineScaleDomain.useConvection = obj.useConvectionData;
            obj.fineScaleDomain = setBoundaries(obj.fineScaleDomain, obj.naturalNodes,...
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
            elseif(strcmp(obj.conductivityDistribution, 'squaredExponential') || ...
                    strcmp(obj.conductivityDistribution, 'ornsteinUhlenbeck') || ...
                    strcmp(obj.conductivityDistribution, 'sincCov') || ...
                    strcmp(obj.conductivityDistribution, 'sincSqCov') || ...
                    strcmp(obj.conductivityDistribution, 'matern'))
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
                        l = obj.conductivityDistributionParams{2};
                    elseif strcmp(obj.conductivityLengthScaleDist, 'lognormal')
                        %First and second parameters are mu and sigma of lognormal dist
                        l = lognrnd(obj.conductivityDistributionParams{2}(1),...
                            obj.conductivityDistributionParams{2}(2));
                        l = [l l];
                    else
                        error('Unknown length scale distribution')
                    end
                    p{i} = genBochnerSamples(l, obj.conductivityDistributionParams{3},...
                        nBochnerBasis, obj.conductivityDistribution);
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
            globalVariationTest = false; %Every microstructure has a common macro-cell
            if globalVariationTest
                obj = obj.getCoarseElement;
                %Conductivity of 10th window is the same in all samples
                cond10 = cond{1}(obj.E(:) == 10);
                E = obj.E(:); %for parfor
            else
                %for parfor
                E = [];
                cond10 = [];
            end
            %Solve finite element model
            disp('Solving finescale problem...')
            tic
            Tf = zeros(obj.fineScaleDomain.nNodes, obj.nSets(nSet));
            D{1} = zeros(2, 2, obj.fineScaleDomain.nEl);
            D = repmat(D, obj.nSets(nSet), 1);
            %To avoid broadcasting overhead
            domain = obj.fineScaleDomain;
            useConv = obj.useConvectionData;
            bcMean = str2num(obj.boundaryConditions);
            bcVariance = obj.boundaryConditionVariance;
            naturalNodes = obj.naturalNodes;
            if(any(bcVariance))
                for i = 1:obj.nSets(nSet)
                    bc{i} = mvnrnd(bcMean, diag(bcVariance));
                end
            else
                bc = [];
            end
            
            parPoolInit(obj.nSets(nSet));
            if useConv
                %Compute coordinates of element centers
                x = .5*(obj.fineScaleDomain.cum_lElX(1:(end - 1)) + obj.fineScaleDomain.cum_lElX(2:end));
                y = .5*(obj.fineScaleDomain.cum_lElY(1:(end - 1)) + obj.fineScaleDomain.cum_lElY(2:end));
                [X, Y] = meshgrid(x, y);
                %directly clear potentially large arrays
                clear y;
                x = [X(:) Y(:)]';
                clear X Y;
            else
                x = []; %unneeded?
            end
            nElf = obj.fineScaleDomain.nEl;
            adp = obj.advectionDistributionParams;
            addpath('./rom');
            ticBytes(gcp)
            parfor i = 1:obj.nSets(nSet)
                %Conductivity matrix D, only consider isotropic materials here
                if globalVariationTest
                    cond{i}(E == 10) = cond10;
                end
                
                if(any(bcVariance))
                    bcTemperature = @(x) bc{i}(1) + bc{i}(2)*x(1) + bc{i}(3)*x(2) + bc{i}(4)*x(1)*x(2);
                    bcHeatFlux{i}{1} = @(x) -(bc{i}(3) + bc{i}(4)*x);      %lower bound
                    bcHeatFlux{i}{2} = @(y) (bc{i}(2) + bc{i}(4)*y);       %right bound
                    bcHeatFlux{i}{3} = @(x) (bc{i}(3) + bc{i}(4)*x);       %upper bound
                    bcHeatFlux{i}{4} = @(y) -(bc{i}(2) + bc{i}(4)*y);      %left bound
                    
                    domainTemp = domain.setBoundaries(naturalNodes, bcTemperature, bcHeatFlux{i});
                    bcHeatFlux = [];
                else
                    domainTemp = domain;
                    bcHeatFlux = [];
                    bcTemperature = [];
                end
                
                for j = 1:domain.nEl
                    D{i}(:, :, j) =  cond{i}(j)*eye(2);
                end
                if useConv
                    %Random convection field
%                     convectionCoeffs = normrnd(adp(1), adp(2), 1, 5)
                    convectionCoeffs = pi*(randi(adp(1), 1, adp(2)) - adp(1)/2)
                    convFieldArray = zeros(2, nElf);
                    for e = 1:nElf
                        convFieldArray(:, e) = convectionField(x(:, e), convectionCoeffs);
                    end
                    FEMout = heat2d(domainTemp, D{i}, convFieldArray);
                    convField{i} = convFieldArray;
                else
                    FEMout = heat2d(domainTemp, D{i});
                end
                %Store fine temperatures as a vector Tf. Use reshape(Tf(:, i), domain.nElX + 1, domain.nElY + 1)
                %and then transpose result to reconvert it to original temperature field
                Ttemp = FEMout.Tff';
                Tf(:, i) = Ttemp(:);
            end
            tocBytes(gcp)
            disp('FEM systems solved')
            tot_FEM_time = toc
            
            if(nargin > 2)
                disp('saving finescale data to...')
                disp(savepath)
                cond = cell2mat(cond);
                if exist('convField')
                    if(any(bcVariance))
                        %partial loading only for -v7.3
                        save(strcat(savepath, ''), 'cond', 'Tf', 'convField', 'bc', '-v7.3')
                    else
                        save(strcat(savepath, ''), 'cond', 'Tf', 'convField', '-v7.3')
                    end
                else
                    if(any(bcVariance))
                        save(strcat(savepath, ''), 'cond', 'Tf', 'bc', '-v7.3')    %partial loading only for -v7.3
                    else
                        save(strcat(savepath, ''), 'cond', 'Tf', '-v7.3')    %partial loading only for -v7.3
                    end
                end
                disp('done')
            end
        end
        
        function obj = genFineScaleDataPath(obj)
            volFrac = obj.conductivityDistributionParams{1};
            sigma_f2 = obj.conductivityDistributionParams{3};
            obj.fineScaleDataPath = strcat(obj.fineScaleDataPath,...
                'systemSize=', num2str(obj.nElFX), 'x', num2str(obj.nElFY), '/');
            %Type of conductivity distribution
            if(strcmp(obj.conductivityDistribution, 'squaredExponential') || ...
                    strcmp(obj.conductivityDistribution, 'ornsteinUhlenbeck') || ...
                    strcmp(obj.conductivityDistribution, 'sincCov') || ...
                    strcmp(obj.conductivityDistribution, 'sincSqCov') || ...
                    strcmp(obj.conductivityDistribution, 'matern'))
                if strcmp(obj.conductivityLengthScaleDist, 'delta')
                    if(obj.conductivityDistributionParams{2}(1) == obj.conductivityDistributionParams{2}(2))
                        corrLength = obj.conductivityDistributionParams{2}(1);
                    else
                        corrLength = obj.conductivityDistributionParams{2};
                    end
                    obj.fineScaleDataPath = strcat(obj.fineScaleDataPath,...
                        obj.conductivityDistribution, '/', 'l=',...
                        num2str(corrLength), '_sigmafSq=', num2str(sigma_f2),...
                        '/volumeFraction=', num2str(volFrac), '/', 'locond=',...
                        num2str(obj.lowerConductivity), '_upcond=', num2str(obj.upperConductivity),...
                        '/', 'BCcoeffs=', obj.boundaryConditions, '/');
                elseif strcmp(obj.conductivityLengthScaleDist, 'lognormal')
                    corrLength1 = obj.conductivityDistributionParams{2}(1);
                    corrLength2 = obj.conductivityDistributionParams{2}(2);
                    obj.fineScaleDataPath = strcat(obj.fineScaleDataPath,...
                        obj.conductivityDistribution, '/', 'l=lognormal_mu=',...
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
            
            if(any(obj.boundaryConditionVariance))
                obj.fineScaleDataPath = strcat(obj.fineScaleDataPath, 'BCvar=[',...
                    num2str(obj.boundaryConditionVariance), ']/');
            end
            
            %THIS NEEDS TO BE GENERALIZED! GIVE ADVECTION FIELD PARAMETERS IN FILE NAME!!!
            if obj.useConvectionData
                obj.fineScaleDataPath = strcat(obj.fineScaleDataPath, 'convection=' , ...
                    num2str(obj.advectionDistributionParams), '/')
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
            obj.coarseScaleDomain.useConvection = obj.useConvection;
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
            obj.outputMean = Tf_true_mean;
            obj.outputVariance = Tf_true_var;
            obj.meanOutputVariance = mean(Tf_true_var);
            toc
            
            %Mean log likelihood
            Tf_true_var(obj.fineScaleDomain.essentialNodes) = NaN;
            nNatNodes = obj.fineScaleDomain.nNodes - numel(obj.fineScaleDomain.essentialNodes);
            Lm = -.5*log(2*pi) - .5 - .5*(1/nNatNodes)*sum(log(Tf_true_var), 'omitnan');
            sv = true;
            if sv
                %     savedir = '~/matlab/data/trueMC/';
                savedir = obj.fineScaleDataPath;
                if ~exist(savedir, 'dir')
                    mkdir(savedir);
                end
                save(strcat(savedir, 'trueMC', '_nSamples=', num2str(nSamples), '.mat'), 'Tf_true_mean', 'Tf_true_var', 'Lm')
            end
        end
        
        function [meanLogLikelihood, err] = estimateLogL(obj, nTrainingData, Tf)
            
            natNodes = true(obj.fineScaleDomain.nNodes, 1);
            natNodes(obj.fineScaleDomain.essentialNodes) = false;
            nNatNodes = sum(natNodes);
            Tf = Tf(natNodes, :);
            meanLogLikelihood = 0;
            meanLogLikelihoodSq = 0;
            converged = false;
            i = 1;
            while(~converged)
                randSamples = randperm(obj.nSets(1));
                randSamples_params = randSamples(1:nTrainingData);
                randSamples_samples = randSamples((nTrainingData + 1):(2*nTrainingData));
                Tftemp = Tf(:, randSamples_params);
                mu_data = mean(Tftemp, 2);
                var_data = var(Tftemp')';
                
                term1 = .5*log(geomean(var_data));
                term2 = .5*mean(mean((Tf(:, randSamples_samples) - mu_data).^2, 2)./var_data);
                
                meanLogLikelihood = ((i - 1)/i)*meanLogLikelihood + (1/i)*(term1 + term2);
                meanLogLikelihoodSq = ((i - 1)/i)*meanLogLikelihoodSq + (1/i)*(term1 + term2)^2;
                if(mod(i, 1000) == 0)
                    i
                    meanLogLikelihood
                    err = sqrt((meanLogLikelihoodSq - meanLogLikelihood^2)/i);
                    relErr = abs(err/meanLogLikelihood)
                    if((relErr < 1e-2 && i > 2e3) || i > 1e5)
                        converged = true;
                    end
                end
                i = i + 1;
            end
            meanLogLikelihood = meanLogLikelihood + .5*log(2*pi);
            meanLogLikelihood = meanLogLikelihood + .87; %remove this!
            err = sqrt((meanLogLikelihoodSq - meanLogLikelihood^2)/i);
            
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
            
            
            if exist(strcat('./data/coarseScaleDomain.mat'), 'file')
                load(strcat('./data/coarseScaleDomain.mat'));
                obj.coarseScaleDomain = coarseScaleDomain;
            else
                warning(strcat('No coarse domain file found.',...
                    'Take boundary conditions from finescale data and regenerate.',...
                    'Please make sure everything is correct!'))
                obj = obj.genCoarseDomain;
            end
            %Load trained params from disk
            disp('Loading optimal parameters from disk...')
            obj.thetaPriorHyperparam = dlmread('./data/thetaPriorHyperparam');
            obj.thetaPriorHyperparam = obj.thetaPriorHyperparam(end, :);
            obj.theta_c.theta = dlmread('./data/theta');
            obj.theta_c.theta = obj.theta_c.theta(end, :)';
            obj.theta_c.Sigma = dlmread('./data/sigma');
            if(numel(obj.theta_c.Sigma) == obj.coarseScaleDomain.nEl)
                obj.theta_c.Sigma = diag(obj.theta_c.Sigma(end, :));
            else
                obj.theta_c.Sigma = diag(obj.theta_c.Sigma(end, :));
            end
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
                disp('done')
            end
        end
        
        function obj = M_step(obj)
            disp('M-step: find optimal params...')
            %Optimal S (decelerated convergence)
            lowerBoundS = eps;
            obj.theta_cf.S = (1 - obj.mix_S)*obj.varExpect_p_cf_exp_mean...
                + obj.mix_S*obj.theta_cf.S + lowerBoundS*ones(obj.fineScaleDomain.nNodes, 1);

            if obj.free_W
                obj.mean_TcTcT(obj.coarseScaleDomain.essentialNodes, :) = [];
                obj.mean_TcTcT(:, obj.coarseScaleDomain.essentialNodes) = [];
                if isempty(obj.fineScaleDomain.essentialNodes)
                    warning('No essential nodes stored in fineScaleDomain. Setting first node to be essential.')
                    obj.fineScaleDomain.essentialNodes = 1;
                end
                obj.mean_TfTcT(obj.fineScaleDomain.essentialNodes, :) = [];
                obj.mean_TfTcT(:, obj.coarseScaleDomain.essentialNodes) = [];
                W_temp = obj.mean_TfTcT/obj.mean_TcTcT;
                
                natNodesFine = 1:obj.fineScaleDomain.nNodes;
                natNodesFine(obj.fineScaleDomain.essentialNodes) = [];
                natNodesCoarse = 1:obj.coarseScaleDomain.nNodes;
                natNodesCoarse(obj.coarseScaleDomain.essentialNodes) = [];
                obj.theta_cf.W(natNodesFine, natNodesCoarse) = W_temp;
            end
            
            obj.theta_cf.Sinv = sparse(1:obj.fineScaleDomain.nNodes,...
                1:obj.fineScaleDomain.nNodes, 1./obj.theta_cf.S);
            obj.theta_cf.Sinv_vec = 1./obj.theta_cf.S;
            obj.theta_cf.WTSinv = obj.theta_cf.W'*obj.theta_cf.Sinv;        %Precomputation for efficiency
            
            %optimal theta_c and sigma
            Sigma_old = obj.theta_c.Sigma;
            theta_old = obj.theta_c.theta;
            
            obj = obj.updateTheta_c;

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
            sumPhiTSigmaInvXmeanOriginal = 0;
            SigmaInv = obj.theta_c.SigmaInv;
            SigmaInvXMean = SigmaInv*obj.XMean;
            sumPhiTSigmaInvPhi = 0;
            sumPhiTSigmaInvPhiOriginal = 0;
            if obj.useConvection
                PhiThetaMat = zeros(3*obj.coarseScaleDomain.nEl, obj.nTrain);
            else
                PhiThetaMat = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
            end
            
            for n = 1:obj.nTrain
                sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + obj.designMatrix{n}'*SigmaInvXMean(:, n);
                sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + obj.designMatrix{n}'*SigmaInv*obj.designMatrix{n};
                PhiThetaMat(:, n) = obj.designMatrix{n}*obj.theta_c.theta;
            end
            
            stabilityParam = 1e-2;    %for stability in matrix inversion
            if(strcmp(obj.thetaPriorType, 'adaptiveGaussian') || strcmp(obj.thetaPriorType, 'RVM') || ...
                    strcmp(obj.thetaPriorType, 'sharedRVM'))
                %Find prior hyperparameter by max marginal likelihood
                converged = false;
                iter = 0;
                if strcmp(obj.thetaPriorType, 'adaptiveGaussian')
                    obj.thetaPriorHyperparam = 1;
                elseif(strcmp(obj.thetaPriorType, 'RVM') || strcmp(obj.thetaPriorType, 'sharedRVM'))
                    if(numel(obj.thetaPriorHyperparam) ~= dim_theta)
                        warning('resizing theta hyperparam')
                        obj.thetaPriorHyperparam = 1e-4*ones(dim_theta, 1);
                    end
%                     lambda_start = [obj.thetaPriorHyperparam (1:100)']
%                     obj.thetaPriorHyperparam = 1e4*ones(dim_theta, 1);
					nElc = obj.coarseScaleDomain.nEl;
					nFeatures = dim_theta/nElc; %for shared RVM
                end
                while(~converged)
                    if strcmp(obj.thetaPriorType, 'adaptiveGaussian')
                        SigmaTilde = inv(sumPhiTSigmaInvPhi + (obj.thetaPriorHyperparam + stabilityParam)*I);
                        muTilde = SigmaTilde*sumPhiTSigmaInvXmean;
                        theta_prior_hyperparam_old = obj.thetaPriorHyperparam;
                        obj.thetaPriorHyperparam = dim_theta/(muTilde'*muTilde + trace(SigmaTilde));
                    elseif(strcmp(obj.thetaPriorType, 'RVM') || strcmp(obj.thetaPriorType, 'sharedRVM'))
                        SigmaTilde = inv(sumPhiTSigmaInvPhi + diag(obj.thetaPriorHyperparam));
%                         muTilde = SigmaTilde*sumPhiTSigmaInvXmean;
%                         muTilde = obj.theta_c.theta;
                        muTilde = (sumPhiTSigmaInvPhi + diag(obj.thetaPriorHyperparam))\sumPhiTSigmaInvXmean;
                        theta_prior_hyperparam_old = obj.thetaPriorHyperparam;
                        if strcmp(obj.thetaPriorType, 'RVM')
%                             gamma = 1 - obj.thetaPriorHyperparam.*diag(SigmaTilde);
%                             gamma(gamma <= 0) = eps;
                            obj.thetaPriorHyperparam = 1./(muTilde.^2 + diag(SigmaTilde));
%                             obj.thetaPriorHyperparam = gamma./(muTilde.^2 + 1e-10);
                        elseif strcmp(obj.thetaPriorType, 'sharedRVM')
                            muTildeSq = muTilde.^2;
                            varTilde = diag(SigmaTilde);
                            lambdaInv = (1/nElc)*(sum(reshape(muTildeSq, nFeatures, nElc), 2) + ...
                                sum(reshape(varTilde, nFeatures, nElc), 2));
                            obj.thetaPriorHyperparam = repmat(1./lambdaInv, nElc, 1);
                        end
                        obj.thetaPriorHyperparam = obj.thetaPriorHyperparam + stabilityParam;
                    end
                    crit = norm(1./obj.thetaPriorHyperparam - 1./theta_prior_hyperparam_old)/...
                            norm(1./obj.thetaPriorHyperparam);
                    if(crit < 1e-5 || iter >= 10)
                        converged = true;
                    elseif(any(~isfinite(obj.thetaPriorHyperparam)) || any(obj.thetaPriorHyperparam <= 0))
                        converged = true;
                        muTilde.^2
                        obj.thetaPriorHyperparam
                        obj.thetaPriorHyperparam = ones(dim_theta, 1);
                        warning('Gaussian hyperparameter precision is negative or not a number. Setting it to 1.')
                    end
                    iter = iter + 1;
                end
%                 lambda_end = [obj.thetaPriorHyperparam (1:100)']
            end
            
            linsolveOpts.SYM = true;
            linsolveOpts.POSDEF = true;
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
                elseif(strcmp(obj.thetaPriorType, 'gaussian') || strcmp(obj.thetaPriorType, 'adaptiveGaussian'))
                    sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + (obj.thetaPriorHyperparam(1) + stabilityParam)*I;
                elseif(strcmp(obj.thetaPriorType, 'RVM') || strcmp(obj.thetaPriorType, 'sharedRVM'))
                    sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + diag(obj.thetaPriorHyperparam);
                elseif strcmp(obj.thetaPriorType, 'none')
                else
                    error('Unknown prior on theta_c')
                end
                
                if (strcmp(obj.thetaPriorType, 'gaussian') || strcmp(obj.thetaPriorType, 'RVM') ||...
                        strcmp(obj.thetaPriorType, 'none') || strcmp(obj.thetaPriorType, 'adaptiveGaussian') || ...
                        strcmp(obj.thetaPriorType, 'sharedRVM'))
                    theta_temp = sumPhiTSigmaInvPhi\sumPhiTSigmaInvXmean;
                    converged = true;   %is this true? we do not need to iteratively maximize theta
                else
%                     theta_temp = U*((U*sumPhiTSigmaInvPhi*U + I)\U)*sumPhiTSigmaInvXmean;
%                     A = linsolve((U*sumPhiTSigmaInvPhi*U + I), U, linsolveOpts);
                    theta_temp = U*linsolve((U*sumPhiTSigmaInvPhi*U + I), U, linsolveOpts)*sumPhiTSigmaInvXmean;
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
                theta= theta_temp;
                
                if obj.useConvection
                    PhiThetaMat = zeros(3*obj.coarseScaleDomain.nEl, obj.nTrain);
                else
                    PhiThetaMat = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
                end
                for n = 1:obj.nTrain
                    PhiThetaMat(:, n) = obj.designMatrix{n}*theta;
                end
                if(obj.EM_iterations < obj.fixSigmaInit)
                    sigma_prior_type = 'delta';
                else
                    sigma_prior_type = obj.sigmaPriorType;
                end
                if strcmp(sigma_prior_type, 'none')
                    if obj.useConvection
                        Sigma = sparse(1:3*obj.coarseScaleDomain.nEl, 1:3*obj.coarseScaleDomain.nEl,...
                        mean(obj.XSqMean - 2*(PhiThetaMat.*obj.XMean) + PhiThetaMat.^2, 2));
                    else
                        if obj.theta_c.full_Sigma
                            sumPhiThetaPhiTThetaT = 0;
                            sumPhiThetaXT = 0;
                            sumXXT = 0;
                            for n = 1:obj.nTrain
                                sumPhiThetaPhiTThetaT = sumPhiThetaPhiTThetaT +...
                                    PhiThetaMat(:, n)*PhiThetaMat(:, n)';
                                sumPhiThetaXT = sumPhiThetaXT + PhiThetaMat(:, n)*obj.XMean(:, n)';
                                sumXXT = sumXXT + obj.XMean(:, n)*obj.XMean(:, n)';
                            end
                            Sigma = diag(mean(obj.XSqMean, 2)) + (sumXXT - diag(diag(sumXXT)))/obj.nTrain +...
                                (sumPhiThetaPhiTThetaT/obj.nTrain) - (sumPhiThetaXT + sumPhiThetaXT')/obj.nTrain;
                        else
                            Sigma = sparse(1:obj.coarseScaleDomain.nEl, 1:obj.coarseScaleDomain.nEl,...
                                mean(obj.XSqMean - 2*(PhiThetaMat.*obj.XMean) + PhiThetaMat.^2, 2));
                        end
                    end
%                     Sigma(Sigma < 0) = eps; %for numerical stability
                    %Variances must be positive
                    Sigma(logical(eye(size(Sigma)))) = abs(Sigma(logical(eye(size(Sigma)))));
                    
                
                    %sum_i Phi_i^T Sigma^-1 <X^i>_qi
                    sumPhiTSigmaInvXmean = 0;
                    %Only valid for diagonal Sigma
%                     s = diag(Sigma);
%                     SigmaInv = sparse(diag(1./s));
                    SigmaInv = inv(Sigma);
%                     SigmaInvXMean = SigmaInv*obj.XMean;
                    SigmaInvXMean = Sigma\obj.XMean;
                    sumPhiTSigmaInvPhi = 0;
                    
                    for n = 1:obj.nTrain
                        sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean + obj.designMatrix{n}'*SigmaInvXMean(:, n);
                        sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi + obj.designMatrix{n}'*(Sigma\obj.designMatrix{n});
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
            if(any(eig(Sigma) <= 0))
                pause
            end
            obj.theta_c.Sigma = Sigma;
            obj.theta_c.SigmaInv = SigmaInv;
            obj.thetaPriorHyperparam = obj.thetaPriorHyperparam;
%             noPriorSigma = mean(obj.XSqMean - 2*(PhiThetaMat.*obj.XMean) + PhiThetaMat.^2, 2);
%             save('./data/noPriorSigma.mat', 'noPriorSigma');
            
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
            if obj.theta_c.full_Sigma
                diag_sigma = diag(obj.theta_c.Sigma)
            end
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
        
        function [lambda_theta_c, lambda_log_s2, lambda_log_sigma2] = laplaceApproximation(obj)
            %Computes parameter precisions based on second derivatives of posterior lower bound F
            
            %p_cf
            %precision of variances s_n^2
            lambda_log_s2 = .5*obj.nTrain*ones(1, obj.fineScaleDomain.nNodes);
            
            %p_c
            %precision of variances sigma_k^2
            if obj.useConvection
                lambda_log_sigma2 = .5*ones(1, 3*obj.coarseScaleDomain.nEl);
            else
                lambda_log_sigma2 = .5*ones(1, obj.coarseScaleDomain.nEl);
            end
            
            %precision on the theta_c's
            lambda_theta_c = zeros(length(obj.theta_c.theta), length(obj.theta_c.theta));
            if isempty(obj.designMatrix)
                load('./persistentData/trainDesignMatrix.mat');
                obj.designMatrix = designMatrix;
            end
            for n = 1:obj.nTrain
                lambda_theta_c = lambda_theta_c + ...
                    obj.designMatrix{n}'*(obj.theta_c.Sigma\obj.designMatrix{n});
            end
            %To ensure symmetry on machine precision; this shouldn't change anything
            lambda_theta_c = .5*(lambda_theta_c + lambda_theta_c');
            
            %Contribution from prior; the contribution from a Laplacian prior is 0
            if strcmp(obj.thetaPriorType, 'RVM')
                lambda_theta_c = lambda_theta_c + diag(obj.thetaPriorHyperparam);
            elseif(strcmp(obj.thetaPriorType, 'gaussian') || strcmp(obj.thetaPriorType, 'adaptiveGaussian'))
                lambda_theta_c = lambda_theta_c + obj.thetaPriorHyperparam*eye(length(obj.theta_c.theta));
            elseif strcmp(obj.thetaPriorType, 'hierarchical_laplace')
                warning('Hessian of Laplace distribution is ill-defined. Ignoring contribution from prior.')
            elseif strcmp(obj.thetaPriorType, 'hierarchical_gamma')
                lambda_theta_c = lambda_theta_c + ...
                    (obj.thetaPriorHyperparam(1) + .5)*...
                    diag(1./(obj.thetaPriorHyperparam(2) + .5*obj.theta_c.theta.^2) - (obj.theta_c.theta.^2)./...
                    ((obj.thetaPriorHyperparam(2) + .5*obj.theta_c.theta.^2).^2));
            end
        end
        
        function obj = predict(obj, mode, boundaryConditions)
            %Function to predict finescale output from generative model
            
            if(nargin < 2)
                mode = 'test';
            end
            
            if(nargin > 2)
                %Predict on different boundary conditions
                obj = obj.setBoundaryConditions(boundaryConditions);
                
                %Set up coarseScaleDomain; must be done after boundary conditions are set up
                obj = obj.genCoarseDomain;
            end
                
            %Load test file
            if strcmp(mode, 'self')
                %Self-prediction on training set
                if obj.useKernels
                    error('Self-prediction using kernel regression not yet implemented')
                end
                Tf = obj.trainingDataMatfile.Tf(:, obj.nStart:(obj.nStart + obj.nTrain - 1));
                tempVars = whos(obj.trainingDataMatfile);
                bcVar = ismember('bc', {tempVars.name});
            else
                Tf = obj.testDataMatfile.Tf(:, obj.testSamples);
                tempVars = whos(obj.testDataMatfile);
                bcVar = ismember('bc', {tempVars.name});
            end
            obj = obj.loadTrainedParams;
            if strcmp(mode, 'self')
                obj = obj.computeDesignMatrix('train');
                designMatrixPred = obj.designMatrix;
            else
                obj = obj.computeDesignMatrix('test');
                designMatrixPred = obj.testDesignMatrix;
            end
            
            %% Sample from p_c
            disp('Sampling from p_c...')
            if strcmp(mode, 'self')
                nTest = obj.nTrain;
            else
                nTest = numel(obj.testSamples);
            end
            
            %short hand notation/ avoiding broadcast overhead
            nElc = obj.coarseScaleDomain.nEl;
            nSamples = obj.nSamples_p_c;
            useConv = obj.useConvection;
%             bcVar = any(obj.boundaryConditionVariance);
            nFineNodes = obj.fineScaleDomain.nNodes;
            
            if useConv
                Xsamples = zeros(3*nElc, nSamples, nTest);
                convectionField{1} = zeros(2, nElc, nSamples);
            else
                Xsamples = zeros(nElc, nSamples, nTest);
                convectionField{1} = [];
            end
            convectionField = repmat(convectionField, nTest, 1);
            LambdaSamples{1} = zeros(nElc, nSamples);
            LambdaSamples = repmat(LambdaSamples, nTest, 1);
            obj.meanEffCond = zeros(nElc, nTest);
            
            if obj.useLaplaceApproximation
                [precisionTheta, precisionLogS, precisionLogSigma] = obj.laplaceApproximation;
                SigmaTheta = inv(precisionTheta) + eps*eye(numel(obj.theta_c.theta));
%                 stdLogS = sqrt(1./precisionLogS)';
%                 stdLogSigma = sqrt(1./precisionLogSigma);
                stdLogS = 0;
                stdLogSigma = 0;
            else
                stdLogS = [];   %for parfor
            end
            
            for i = 1:nTest
                %Samples from p_c
                if obj.useLaplaceApproximation
                    %First sample theta from Laplace approx, then sample X
                    theta = mvnrnd(obj.theta_c.theta', SigmaTheta, nSamples)';
                    for j = 1:nSamples
                        Sigma2 = exp(normrnd(log(diag(obj.theta_c.Sigma))', stdLogSigma));
                        Sigma2 = diag(obj.theta_c.Sigma);
                        Xsamples(:, j, i) = mvnrnd((designMatrixPred{i}*theta(:, j))', Sigma2)';
                    end
                else
                    Xsamples(:, :, i) = mvnrnd((designMatrixPred{i}*obj.theta_c.theta)',...
                        obj.theta_c.Sigma, nSamples)';
                end
                %Conductivities
                LambdaSamples{i} = conductivityBackTransform(Xsamples(1:nElc, :, i),...
                    obj.conductivityTransformation);
                if(strcmp(obj.conductivityTransformation.type, 'log'))
                    obj.meanEffCond(:, i) = exp(obj.testDesignMatrix{i}*obj.theta_c.theta + .5*diag(obj.theta_c.Sigma));
                else
                    obj.meanEffCond(:, i) = mean(LambdaSamples{i}, 2);
                end
                if obj.useConvection
                    %Convection field
                    for j = 1:nSamples
                        convectionField{i}(:, :, j) = [Xsamples((nElc + 1):(2*nElc), j, i)';...
                            Xsamples((2*nElc + 1):(3*nElc), j, i)'];
                    end
                end
            end
            disp('done')
            
            %% Run coarse model and sample from p_cf
            disp('Solving coarse model and sample from p_cf...')
            TfMeanArray{1} = zeros(obj.fineScaleDomain.nNodes, 1);
            TfMeanArray = repmat(TfMeanArray, nTest, 1);
            TfVarArray = TfMeanArray;
            Tf_sq_mean = TfMeanArray;
            
            if(bcVar)
                %Set coarse domain for data with different boundary conditions
                nX = obj.coarseScaleDomain.nElX;
                nY = obj.coarseScaleDomain.nElY;
                if strcmp(mode, 'self')
                    bc = obj.trainingDataMatfile.bc;
                else
                    bc = obj.testDataMatfile.bc;
                end
                for j = 1:nTest
                    if strcmp(mode, 'self')
                        i = obj.trainingSamples(j);
                    else
                        i = obj.testSamples(j);
                    end
                    bcT = @(x) bc{i}(1) + bc{i}(2)*x(1) + bc{i}(3)*x(2) + bc{i}(4)*x(1)*x(2);
                    bcQ{1} = @(x) -(bc{i}(3) + bc{i}(4)*x);      %lower bound
                    bcQ{2} = @(y) (bc{i}(2) + bc{i}(4)*y);       %right bound
                    bcQ{3} = @(x) (bc{i}(3) + bc{i}(4)*x);       %upper bound
                    bcQ{4} = @(y) -(bc{i}(2) + bc{i}(4)*y);      %left bound
                    cd(i) = obj.coarseScaleDomain;
                    cd(i) = cd(i).setBoundaries([2:(2*nX + 2*nY)], bcT, bcQ);
                end
            else
                cd = obj.coarseScaleDomain;
            end
            t_cf = obj.theta_cf;
            lapAp = obj.useLaplaceApproximation;
            %             t_c = obj.theta_c;
            natNodes = true(obj.fineScaleDomain.nNodes, 1);
            natNodes(obj.fineScaleDomain.essentialNodes) = false;
            nNatNodes = sum(natNodes)
            addpath('./heatFEM');
            parfor j = 1:nTest
                if bcVar
                    coarseDomain = cd(j);
                else
                    coarseDomain = cd;
                end
                for i = 1:nSamples
                    D = zeros(2, 2, coarseDomain.nEl);
                    for e = 1:coarseDomain.nEl
                        D(:, :, e) = LambdaSamples{j}(e, i)*eye(2);
                    end
                    if useConv
                        FEMout = heat2d(coarseDomain, D, convectionField{j}(:, :, i))
                    else
                        FEMout = heat2d(coarseDomain, D);
                    end
                    Tctemp = FEMout.Tff';
                    
                    %sample from p_cf
                    mu_cf = t_cf.mu + t_cf.W*Tctemp(:);
                    %only for diagonal S!!
                    %Sequentially compute mean and <Tf^2> to save memory
                    TfMeanArray{j} = ((i - 1)/i)*TfMeanArray{j} + (1/i)*mu_cf;  %U_f-integration can be done analyt.
                    Tf_sq_mean{j} = ((i - 1)/i)*Tf_sq_mean{j} + (1/i)*mu_cf.^2;
                end
                if lapAp
%                     S = exp(normrnd(log(t_cf.S), stdLogS));
                    S = t_cf.S;
                    Tf_sq_mean{j} = Tf_sq_mean{j} + S;
                else
                    Tf_sq_mean{j} = Tf_sq_mean{j} + t_cf.S;
                end
                Tf_var = abs(Tf_sq_mean{j} - TfMeanArray{j}.^2);  %abs to avoid negative variance due to numerical error
                meanTf_meanMCErr = mean(sqrt(Tf_var/nSamples))
                TfVarArray{j} = Tf_var;
                
                meanMahaErrTemp{j} = mean(sqrt(abs((1./(Tf_var)).*(Tf(:, j) - TfMeanArray{j}).^2)));
                sqDist{j} = (Tf(:, j) - TfMeanArray{j}).^2;
                meanSqDistTemp{j} = mean(sqDist{j});
                
                Tf_var_nat = Tf_var(natNodes);
                logLikelihood{j} = -.5*nNatNodes*log(2*pi) - .5*sum(log(Tf_var_nat), 'omitnan') - ...
                    .5*sum(sqDist{j}(natNodes)./Tf_var_nat, 'omitnan');
                logPerplexity{j} = -(1/(nNatNodes))*logLikelihood{j};
            end
            
            obj.meanPredMeanOutput = mean(cell2mat(TfMeanArray'), 2);
            obj.meanMahalanobisError = mean(cell2mat(meanMahaErrTemp));
            obj.meanSquaredDistanceField = mean(cell2mat(sqDist), 2);
            obj.meanSquaredDistance = mean(cell2mat(meanSqDistTemp));
            meanSqDistSq = mean(cell2mat(meanSqDistTemp).^2);
            obj.meanSquaredDistanceError = sqrt((meanSqDistSq - obj.meanSquaredDistance^2)/nTest);
            obj.meanLogLikelihood = mean(cell2mat(logLikelihood))/nNatNodes;
            obj.meanLogPerplexity = mean(cell2mat(logPerplexity));
            obj.meanPerplexity = exp(obj.meanLogPerplexity);
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
                    cond = obj.testDataMatfile.cond(:, obj.testSamples(1):(obj.testSamples(1) + 5));
                end
                %to use same color scale
                cond = ((min_Tf - max_Tf)/(min(min(cond)) - max(max(cond))))*cond + max_Tf - ...
                    ((min_Tf - max_Tf)/(min(min(cond)) - max(max(cond))))*max(max(cond));
                for i = pstart:(pstart + 5)
                    subplot(2, 3, j)
                    Tf_i_min = min(Tf(:, i))
                    s(j, 1) = surf(reshape(Tf(:, i) - Tf_i_min, (obj.nElFX + 1), (obj.nElFY + 1)));
                    s(j, 1).LineStyle = 'none';
                    hold on;
                    s(j, 2) = surf(reshape(TfMeanArray{i} - Tf_i_min, (obj.nElFX + 1), (obj.nElFY + 1)));
                    s(j, 2).LineStyle = 'none';
                    s(j, 2).FaceColor = 'b';
                    s(j, 3) = surf(reshape(TfMeanArray{i} - Tf_i_min, (obj.nElFX + 1), (obj.nElFY + 1)) +...
                        sqrt(reshape(TfVarArray{i}, (obj.nElFX + 1), (obj.nElFY + 1))));
                    s(j, 3).LineStyle = 'none';
                    s(j, 3).FaceColor = [.85 .85 .85];
                    s(j, 4) = surf(reshape(TfMeanArray{i} - Tf_i_min, (obj.nElFX + 1), (obj.nElFY + 1)) -...
                        sqrt(reshape(TfVarArray{i}, (obj.nElFX + 1), (obj.nElFY + 1))));
                    s(j, 4).LineStyle = 'none';
                    s(j, 4).FaceColor = [.85 .85 .85];
                    ax = gca;
                    ax.FontSize = 30;
                    im(j) = imagesc(reshape(cond(:, i), obj.fineScaleDomain.nElX, obj.fineScaleDomain.nElY));
                    xticks([0 64 128 192 256]);
                    yticks([0 64 128 192 256]);
%                     zticks(100:100:800)
                    xticklabels({});
                    yticklabels({});
                    zticklabels({});
                    axis tight;
                    axis square;
                    box on;
                    view(-60, 15)
                    zlim([min_Tf max_Tf]);
                    caxis([min_Tf max_Tf]);
                    j = j + 1;
                end
%                 print(f, './predictions', '-dpng', '-r300')
            end
        end

        function [predMean, predSqMean] = randThetaPredict(obj, mode, nSamplesTheta, boundaryConditions)
            %Function to predict finescale output from generative model
            
            if(nargin < 2)
                mode = 'test';
            end
            
            if(nargin > 3)
                %Predict on different boundary conditions
                obj = obj.setBoundaryConditions(boundaryConditions);
                
                %Set up coarseScaleDomain; must be done after boundary conditions are set up
                obj = obj.genCoarseDomain;
            end
                
            %Load test file
            if strcmp(mode, 'self')
                %Self-prediction on training set
                assert(~obj.useKernels, 'Self-prediction using kernel regression not yet implemented')
                Tf = obj.trainingDataMatfile.Tf(:, obj.nStart:(obj.nStart + obj.nTrain - 1));
                tempVars = whos(obj.trainingDataMatfile);
                bcVar = ismember('bc', {tempVars.name});
            else
                Tf = obj.testDataMatfile.Tf(:, obj.testSamples);
                tempVars = whos(obj.testDataMatfile);
                bcVar = ismember('bc', {tempVars.name});
            end
            obj = obj.loadTrainedParams;
            if strcmp(mode, 'self')
                obj = obj.computeDesignMatrix('train');
                designMatrixPred = obj.designMatrix;
            else
                obj = obj.computeDesignMatrix('test');
                designMatrixPred = obj.testDesignMatrix;
            end
            
            %% Sample from p_c
            disp('Sampling from p_c...')
            if strcmp(mode, 'self')
                nTest = obj.nTrain;
            else
                nTest = numel(obj.testSamples);
            end
            
            %short hand notation/ avoiding broadcast overhead
            nElc = obj.coarseScaleDomain.nEl;
            nSamplesLambda_c = obj.nSamples_p_c;
            
            XsamplesTemp = zeros(nElc, nSamplesLambda_c);
            
            LambdaSamples{1} = zeros(nElc, nSamplesLambda_c);
            LambdaSamples = repmat(LambdaSamples, nTest, nSamplesTheta);
            obj.meanEffCond = zeros(nElc, nTest);
            
            if obj.useLaplaceApproximation
                precisionTheta = obj.laplaceApproximation;
                SigmaTheta = inv(precisionTheta) + eps*eye(numel(obj.theta_c.theta));
            else
                SigmaTheta = zeros(numel(obj.theta_c.theta));
                nSamplesTheta = 1;
            end
            
            for n = 1:nTest
                %Samples from p_c
                %First sample theta from Laplace approx, then sample X
                theta = mvnrnd(obj.theta_c.theta', SigmaTheta, nSamplesTheta)';
                for t = 1:nSamplesTheta
                    for k = 1:nSamplesLambda_c
                        Sigma2 = obj.theta_c.Sigma;
                        XsamplesTemp(:, k) = mvnrnd((designMatrixPred{n}*theta(:, t))', Sigma2)';
                    end
                %Conductivities
                LambdaSamples{n, t} = conductivityBackTransform(XsamplesTemp,...
                    obj.conductivityTransformation);
                end
            end
            clear Xsamples; %save memory
            disp('done')
            
            %% Run coarse model and sample from p_cf
            disp('Solving coarse model and sample from p_cf...')
            TfMean{1} = zeros(obj.fineScaleDomain.nNodes, nTest);
            TfMean = repmat(TfMean, nSamplesTheta, 1);
            TfVar = TfMean;
            Tf_sq_mean = TfMean;
            
            if(bcVar)
                %Set coarse domain for data with different boundary conditions
                nX = obj.coarseScaleDomain.nElX;
                nY = obj.coarseScaleDomain.nElY;
                if strcmp(mode, 'self')
                    bc = obj.trainingDataMatfile.bc;
                else
                    bc = obj.testDataMatfile.bc;
                end
                for n = 1:nTest
                    if strcmp(mode, 'self')
                        i = obj.trainingSamples(n);
                    else
                        i = obj.testSamples(n);
                    end
                    bcT = @(x) bc{i}(1) + bc{i}(2)*x(1) + bc{i}(3)*x(2) + bc{i}(4)*x(1)*x(2);
                    bcQ{1} = @(x) -(bc{i}(3) + bc{i}(4)*x);      %lower bound
                    bcQ{2} = @(y) (bc{i}(2) + bc{i}(4)*y);       %right bound
                    bcQ{3} = @(x) (bc{i}(3) + bc{i}(4)*x);       %upper bound
                    bcQ{4} = @(y) -(bc{i}(2) + bc{i}(4)*y);      %left bound
                    cd(n) = obj.coarseScaleDomain;
                    cd(n) = cd(n).setBoundaries([2:(2*nX + 2*nY)], bcT, bcQ);
                end
            else
                cd = obj.coarseScaleDomain;
            end
            t_cf = obj.theta_cf;
            natNodes = true(obj.fineScaleDomain.nNodes, 1);
            natNodes(obj.fineScaleDomain.essentialNodes) = false;
            addpath('./heatFEM');
            parfor t = 1:nSamplesTheta
                for n = 1:nTest
                    if bcVar
                        coarseDomain = cd(n);
                    else
                        coarseDomain = cd;
                    end
                    for i = 1:nSamplesLambda_c
                        D = zeros(2, 2, coarseDomain.nEl);
                        for e = 1:coarseDomain.nEl
                            D(:, :, e) = LambdaSamples{n, t}(e, i)*eye(2);
                        end
                        
                        FEMout = heat2d(coarseDomain, D);
                        Tctemp = FEMout.Tff';
                        
                        %sample from p_cf
                        mu_cf = t_cf.mu + t_cf.W*Tctemp(:);
                        %only for diagonal S!!
                        %Sequentially compute mean and <Tf^2> to save memory
                        TfMean{t}(:, n) = ((i - 1)/i)*TfMean{t}(:, n) + (1/i)*mu_cf;  %U_f-integration can be done analyt.
                        Tf_sq_mean{t}(:, n) = ((i - 1)/i)*Tf_sq_mean{t}(:, n) + (1/i)*mu_cf.^2;
                    end
                    
                    Tf_sq_mean{t}(:, n) = Tf_sq_mean{t}(:, n) + t_cf.S;
                end
                predMean{t} = mean(TfMean{t}, 2);
                predSqMean{t} = mean(Tf_sq_mean{t}, 2);
                TfMean{t} = []; %save memory
                Tf_sq_mean{t} = [];
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

        function [lambdak, xk, ak] = get_coarseElementConductivities(obj, mode, samples)
            %Cuts out conductivity/convection fields from macro-cells
			addpath('./rom');            

            %load finescale conductivity field
            if strcmp(mode, 'train')
                if(nargin < 3)
                    conductivity = obj.trainingDataMatfile.cond(:, obj.trainingSamples);
                    if(nargout > 2)
                        %Save format: (x/y-component, fineElement, sample)
                        dataVars = whos(obj.trainingDataMatfile);
                        if ismember('convField', {dataVars.name})
                            convectionField = obj.trainingDataMatfile.convField(1, obj.trainingSamples);
                        else
                            convectionField = [];
                        end
                    end
                else
                    %used for pca
                    conductivity = obj.trainingDataMatfile.cond(:, samples);
                end
            elseif strcmp(mode, 'test')
                conductivity = obj.testDataMatfile.cond(:, obj.testSamples);
                if(nargout > 2)
                    %Save format: (x/y-component, fineElement, sample)
                    dataVars = whos(obj.testDataMatfile);
                    if ismember('convField', {dataVars.name})
                        convectionField = obj.testDataMatfile.convField(1, obj.trainingSamples);
                    else
                        convectionField = [];
                    end
                end
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
                if(nargout > 2)
                    ak = lambdak;
                end
            end
            
            for s = 1:nData
                %inputs belonging to same coarse element are in the same column of xk. They are ordered in
                %x-direction.
                %Get conductivity fields in coarse cell windows
                %Might be wrong for non-square fine scale domains
                conductivityMat = reshape(conductivity(:, s), obj.fineScaleDomain.nElX,...
                    obj.fineScaleDomain.nElY);
                if(nargout > 2)
                    if ~isempty(convectionField)
                        convectionFieldMatX = reshape(convectionField{s}(1, :), obj.fineScaleDomain.nElX,...
                            obj.fineScaleDomain.nElY);
                        convectionFieldMatY = reshape(convectionField{s}(2, :), obj.fineScaleDomain.nElX,...
                            obj.fineScaleDomain.nElY);
                    end
                end
                for e = 1:obj.coarseScaleDomain.nEl
                    indexMat = (EHold == e);
                    if obj.padding
                        indexMat = padIndexMat(indexMat, obj.padding);
                    end

                    lambdakTemp = conductivityMat.*indexMat;
                    %Cut elements from matrix that do not belong to coarse cell
                    lambdakTemp(~any(lambdakTemp, 2), :) = [];
                    lambdakTemp(:, ~any(lambdakTemp, 1)) = [];
                    lambdak{s, e} = lambdakTemp;
                    if(nargout > 1)
                        xk{s, e} = conductivityTransform(lambdak{s, e}, obj.conductivityTransformation);
                    end
                    
                    if(nargout > 2)
                        %Removing of padding as above is dangerous as ak's might be 0.
                        %Thus, add eps and remove it later on
                        if ~isempty(convectionField)
                            akX = (convectionFieldMatX + eps).*indexMat;
                            akY = (convectionFieldMatY + eps).*indexMat;
                            %Cut elements from matrix that do not belong to coarse cell
                            akX(~any(akX, 2), :) = [];  %Use lambdakTemp to define indices as it is never 0
                            akY(~any(akY, 2), :) = [];
                            akX(:, ~any(akX, 1)) = [];
                            akY(:, ~any(akY, 1)) = [];
                            ak{s, e} = cat(3, akX, akY); %stored as a two-sheets array
                        else
                            ak{s, e} = [];
                        end
                    end
                end
            end
        end
        
        function obj = computeKernelMatrix(obj, mode)
            %Must be called BEFORE setting up any local lode (useLocal, useNeighbor,...)
            disp('Using kernel regression mode.')
            
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
            debug = false; %for debug mode
            tic
            if(obj.loadDesignMatrix && ~recompute && strcmp(mode, 'train'))
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
                dataVars = whos(dataFile);
                if ismember('convField', {dataVars.name})
                    %convField is cell array
                    convectionField = dataFile.convField(1, dataSamples);
%                     convectionField = num2cell(convectionField, 1:2);
                else
                    warning('No finescale convection field stored, setting it to [].')
                    convectionField = [];
                end
                %set feature function handles
                [obj, phi, phiGlobal, phiConvection, phiGlobalConvection] = obj.setFeatureFunctions;
                for j = 1:size(phi, 2)
                    if(j == 1)
                        dlmwrite('./data/features', func2str(phi{1, j}), 'delimiter', '');
                    else
                        dlmwrite('./data/features', func2str(phi{1, j}),...
                            'delimiter', '', '-append');
                    end
                end
                for j = 1:size(phiGlobal, 2)
                    dlmwrite('./data/features', func2str(phiGlobal{1, j}),...
                        'delimiter', '', '-append');
                end
                nFeatureFunctions = size(obj.featureFunctions, 2);
                nGlobalFeatureFunctions = size(obj.globalFeatureFunctions, 2);
                nConvectionFeatureFunctions = size(obj.convectionFeatureFunctions, 2);
                nGlobalConvectionFeatureFunctions = size(obj.globalConvectionFeatureFunctions, 2);
                nTotalFeatures = nFeatureFunctions + nGlobalFeatureFunctions + ...
                    nConvectionFeatureFunctions + nGlobalConvectionFeatureFunctions + obj.latentDim;
%                 phi = obj.featureFunctions;
%                 phiGlobal = obj.globalFeatureFunctions;
%                 phiConvection = obj.convectionFeatureFunctions;
%                 phiGlobalConvection = obj.globalConvectionFeatureFunctions;
                %Open parallel pool
                addpath('./computation')
                parPoolInit(nData);
                if obj.useConvection
                    PhiCell{1} = zeros(3*obj.coarseScaleDomain.nEl, 3*nTotalFeatures);
                    [lambdak, xk, ak] = obj.get_coarseElementConductivities(mode);
                else
                    PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, nFeatureFunctions + nGlobalFeatureFunctions);
                    [lambdak, xk] = obj.get_coarseElementConductivities(mode);
                    ak = [];
                end
                PhiCell = repmat(PhiCell, nData, 1);
                
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
                else
                    latentMu = [];
                end
                
                %avoid broadcasting overhead
                nElc = obj.coarseScaleDomain.nEl;
                nElXf = obj.fineScaleDomain.nElX;
                nElYf = obj.fineScaleDomain.nElY;
                uae = obj.useAutoEnc;
                ld = obj.latentDim;
                uc = obj.useConvection;
                ticBytes(gcp)
                parfor s = 1:nData    %for very cheap features, serial evaluation might be more efficient
                    %inputs belonging to same coarse element are in the same column of xk. They are ordered in
                    %x-direction.
                    
                    %construct conductivity design matrix
                    for i = 1:nElc
                        %local features
                        for j = 1:nFeatureFunctions
                            %only take pixels of corresponding macro-cell as input for features
                            PhiCell{s}(i, j) = phi{i, j}(lambdak{s, i});
                        end
                        %global features
                        for j = 1:nGlobalFeatureFunctions
                            %Take whole microstructure as input for feature function
                            %Might be wrong for non-square fine scale domains
                            conductivityMat = reshape(conductivity{s}, nElXf, nElYf);
                            PhiCell{s}(i, nFeatureFunctions + j) = phiGlobal{i, j}(conductivityMat);
                        end
                        if uae
                            for j = 1:ld
                                PhiCell{s}(i, nFeatureFunctions + nGlobalFeatureFunctions + j) = latentMu(j, i, s);
                            end
                        end
                    end
                    %NEEDS TO BE UNCOMMENTED FOR USE OF CONVECTION FIELD! NOT POSSIBLE WITH PARFOR
%                     if uc
%                         %construct convection design matrix
%                         for i = 1:nElc
%                             %local features
%                             for j = 1:nConvectionFeatureFunctions
%                                 %only take pixels of corresponding macro-cell as input for features
%                                 PhiCell{s}(i, j + nFeatureFunctions + nGlobalFeatureFunctions + ld) =...
%                                     phiConvection{i, j}(ak{s, i});
%                             end
%                             %global features
%                             for j = 1:nGlobalConvectionFeatureFunctions
%                                 %Take whole microstructure as input for feature function
%                                 %Might be wrong for non-square fine scale domains
%                                 convectionMat = reshape(convectionField{s}, 2, nElXf, nElYf);
%                                 PhiCell{s}(i, j + nFeatureFunctions + nGlobalFeatureFunctions...
%                                     + ld + nConvectionFeatureFunctions) =...
%                                     phiGlobalConvection{i, j}(convectionMat);
%                             end
%                         end
%                         %Fill in lower right elements of design matrix. These are predictors for
%                         %convection field A
%                         PhiCell{s}((nElc + 1):(2*nElc), (nTotalFeatures + 1):2*(nTotalFeatures)) =...
%                             PhiCell{s}(1:nElc, 1:nTotalFeatures);
%                         PhiCell{s}((2*nElc + 1):end, (2*nTotalFeatures + 1):end) =...
%                             PhiCell{s}(1:nElc, 1:nTotalFeatures);
%                     else
%                         %do nothing. for parfor
%                     end
                end
                tocBytes(gcp)
                
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
%                 obj.designMatrix = PhiCell;
                %Check for real finite inputs
                for i = 1:nData
                    if(~all(all(all(isfinite(PhiCell{i})))))
                        dataPoint = i
                        [coarseElement, featureFunction] = ind2sub(size(PhiCell{i}),...
                            find(~isfinite(PhiCell{i})))
                        warning('Non-finite design matrix. Setting non-finite component to 0.')
                        PhiCell{i}(~isfinite(PhiCell{i})) = 0;
                    elseif(~all(all(all(isreal(PhiCell{i})))))
                        warning('Complex feature function output:')
                        dataPoint = i
                        [coarseElement, featureFunction] = ind2sub(size(PhiCell{i}),...
                            find(imag(PhiCell{i})))
                        disp('Ignoring imaginary part...')
                        PhiCell{i} = real(PhiCell{i});
                    end
                end
                disp('done')
                
                if strcmp(mode, 'train')
                    obj.designMatrix = PhiCell;
                elseif strcmp(mode, 'test')
                    obj.testDesignMatrix = PhiCell;
                else
                    error('Wrong design matrix computation model')
                end
                
                %Include second order combinations of features
                if(any(any(obj.secondOrderTerms)))
                    obj = obj.secondOrderFeatures(mode);
                end
                if(obj.useKernels)
                    if(strcmp(obj.bandwidthSelection, 'scott') || strcmp(obj.bandwidthSelection, 'silverman'))
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
                            %Scott's rule of thumb
                            if strcmp(obj.mode, 'none')
                                obj.kernelBandwidth = (obj.nTrain*obj.coarseScaleDomain.nEl)^(-1/(nFeatures + 4))*...
                                    featureFunctionStd;
                            elseif strcmp(obj.mode, 'useLocal')
                                obj.kernelBandwidth = obj.nTrain^(-1/(nFeatures + 4))*featureFunctionStd;
                            else
                                error('No rule of thumb implemented for this mode')
                            end
                            if strcmp(obj.bandwidthSelection, 'silverman')
                                obj.kernelBandwidth = obj.kernelBandwidth*(4/(nFeatures + 2))^(1/(nFeatures + 4));
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
                    
                else
                    if strcmp(mode, 'train')
                        PhiCell = obj.designMatrix;
                    elseif strcmp(mode, 'test')
                        PhiCell = obj.testDesignMatrix;
                    else
                        error('Wrong design matrix computation model')
                    end
                    %Normalize design matrices. Do not do this using kernels! Kernel bandwidth is
                    %adjusted instead!
                    if strcmp(obj.featureScaling, 'standardize')
                        obj = obj.standardizeDesignMatrix(mode, PhiCell);
                    elseif strcmp(obj.featureScaling, 'rescale')
                        obj = obj.rescaleDesignMatrix(mode, PhiCell);
                    elseif strcmp(obj.featureScaling, 'normalize')
                        obj = obj.normalizeDesignMatrix(mode, PhiCell);
                    else
                        disp('No feature scaling used...')
                    end
                end
                
                %Design matrix is always stored in its original form. Local modes are applied after
                %loading
                if strcmp(mode, 'train')
                    designMatrix = obj.designMatrix;
                    obj.originalDesignMatrix = obj.designMatrix;
                    save(strcat('./persistentData/', mode, 'DesignMatrix.mat'), 'designMatrix')
                end
            end
            if obj.useKernels
                %This step might be confusing: after storing, we replace the design matrix by the kernel matrix
                if strcmp(mode, 'train')
                    obj.designMatrix = obj.kernelMatrix;
                elseif strcmp(mode, 'test')
                    obj.testDesignMatrix = obj.kernelMatrix;
                else
                    error('wrong mode')
                end
                PhiCell = obj.kernelMatrix;
                obj.kernelMatrix = [];  %to save memory
            end
            %Use specific nonlocality mode
            if strcmp(obj.mode, 'useNeighbor')
                %use feature function information from nearest neighbors
                obj = obj.includeNearestNeighborFeatures(PhiCell, mode);
            elseif strcmp(obj.mode, 'useLocalNeighbor')
                obj = obj.includeLocalNearestNeighborFeatures(PhiCell, mode);
            elseif strcmp(obj.mode, 'useLocalDiagNeighbor')
                obj = obj.includeLocalDiagNeighborFeatures(PhiCell, mode);
            elseif strcmp(obj.mode, 'useDiagNeighbor')
                %use feature function information from nearest and diagonal neighbors
                obj = obj.includeDiagNeighborFeatures(PhiCell, mode);
            elseif strcmp(obj.mode, 'useLocal')
                %Use separate parameters for every macro-cell
                obj = obj.localTheta_c(PhiCell, mode);
            else
                obj.originalDesignMatrix = [];
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
            
            nFeatureFunctions = size(obj.featureFunctions, 2) + size(obj.globalFeatureFunctions, 2);
            if obj.useAutoEnc
                nFeatureFunctions = nFeatureFunctions + obj.latentDim;
            end
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
                if strcmp(mode, 'train')
                    PhiCell{s}(:, 1:nFeatureFunctions) = obj.designMatrix{s};
                elseif strcmp(mode, 'test')
                    PhiCell{s}(:, 1:nFeatureFunctions) = obj.testDesignMatrix{s};
                else
                    error('wrong mode')
                end
                
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
            if strcmp(mode, 'train')
                obj.designMatrix = PhiCell;
            elseif strcmp(mode, 'test')
                obj.testDesignMatrix = PhiCell;
            else
                error('wrong mode');
            end
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

        function obj = standardizeDesignMatrix(obj, mode, designMatrix)
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
            
            %Check if there is a constant feature function
            for i = 1:length(featureFunctionStd)
                if(featureFunctionStd == 0)
                    i
                    featureFunctionStd(i) = obj.featureFunctionMean(i);
                    warning('At least one feature always has the same output. It will be rescaled to one.')
                    break;
                end
            end
            
            %centralize
            for i = 1:numel(designMatrix)
                designMatrix{i} = designMatrix{i} - obj.featureFunctionMean;
            end
            
            %normalize
            for i = 1:numel(designMatrix)
                designMatrix{i} = designMatrix{i}./featureFunctionStd;
            end
            
            %Check for finiteness
            for i = 1:numel(designMatrix)
                if(~all(all(all(isfinite(designMatrix{i})))))
                    warning('Non-finite design matrix. Setting non-finite component to 0.')
                    designMatrix{i}(~isfinite(designMatrix{i})) = 0;
                elseif(~all(all(all(isreal(designMatrix{i})))))
                    warning('Complex feature function output:')
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(designMatrix{i}),...
                        find(imag(designMatrix{i})))
                    disp('Ignoring imaginary part...')
                    designMatrix{i} = real(designMatrix{i});
                end
            end
            if strcmp(mode, 'train')
                obj.designMatrix = designMatrix;
            elseif strcmp(mode, 'test')
                obj.testDesignMatrix = designMatrix;
            else
                error('wrong mode')
            end
            obj.saveNormalization('standardization');
            disp('done')
        end
        
        function obj = normalizeDesignMatrix(obj, mode, designMatrix)
            %Standardize covariates to unit variance
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
            
            %Check if there is a constant feature function
            for i = 1:length(featureFunctionStd)
                if(featureFunctionStd(i) == 0)
                    i
                    featureFunctionStd(i) = obj.featureFunctionMean(i);
                    warning('At least one feature always has the same output. It will be rescaled to one.')
                    break;
                end
            end
            
            %normalize
            for i = 1:numel(designMatrix)
                designMatrix{i} = designMatrix{i}./featureFunctionStd;
            end
            
            %Check for finiteness
            for i = 1:numel(designMatrix)
                if(~all(all(all(isfinite(designMatrix{i})))))
                    warning('Non-finite design matrix. Setting non-finite component to 0.')
                    designMatrix{i}(~isfinite(designMatrix{i})) = 0;
                elseif(~all(all(all(isreal(designMatrix{i})))))
                    warning('Complex feature function output:')
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(designMatrix{i}),...
                        find(imag(designMatrix{i})))
                    disp('Ignoring imaginary part...')
                    designMatrix{i} = real(designMatrix{i});
                end
            end
            if strcmp(mode, 'train')
                obj.designMatrix = designMatrix;
            elseif strcmp(mode, 'test')
                obj.testDesignMatrix = designMatrix;
            else
                error('wrong mode')
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
        
        function obj = rescaleDesignMatrix(obj, mode, designMatrix)
            %Rescale design matrix s.t. outputs are between 0 and 1
            disp('Rescale design matrix...')
            if strcmp(mode, 'test')
                featFuncDiff = obj.featureFunctionMax - obj.featureFunctionMin;
                %to avoid irregularities due to rescaling (if every macro cell has the same feature function output)
                obj.featureFunctionMin(featFuncDiff == 0) = 0;
                featFuncDiff(featFuncDiff == 0) = 1;
                for n = 1:numel(designMatrix)
                    designMatrix{n} = (designMatrix{n} - obj.featureFunctionMin)./(featFuncDiff);
                end
            else
                obj = obj.computeFeatureFunctionMinMax;
                featFuncDiff = obj.featureFunctionMax - obj.featureFunctionMin;
                %to avoid irregularities due to rescaling (if every macro cell has the same feature function output)
                obj.featureFunctionMin(featFuncDiff == 0) = 0;
                featFuncDiff(featFuncDiff == 0) = 1;
                for n = 1:numel(designMatrix)
                    designMatrix{n} = (designMatrix{n} - obj.featureFunctionMin)./(featFuncDiff);
                end
            end
            %Check for finiteness
            for n = 1:numel(designMatrix)
                if(~all(all(all(isfinite(designMatrix{n})))))
                    warning('Non-finite design matrix. Setting non-finite component to 0.')
                    designMatrix{n}(~isfinite(designMatrix{n})) = 0;
                    dataPoint = n
                    [coarseElement, featureFunction] = ind2sub(size(designMatrix{n}),...
                        find(~isfinite(designMatrix{n})))
                elseif(~all(all(all(isreal(designMatrix{n})))))
                    warning('Complex feature function output:')
                    dataPoint = n
                    [coarseElement, featureFunction] = ind2sub(size(designMatrix{n}),...
                        find(imag(designMatrix{n})))
                    disp('Ignoring imaginary part...')
                    designMatrix{n} = real(designMatrix{n});
                end
            end
            if strcmp(mode, 'train')
                obj.designMatrix = designMatrix;
            elseif strcmp(mode, 'test')
                obj.testDesignMatrix = designMatrix;
            else
                error('wrong mode')
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
        
        function obj = includeNearestNeighborFeatures(obj, designMatrix, mode)
            %Includes feature function information of neighboring cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest neighbor feature function information...')
            nFeatureFunctionsTotal = size(designMatrix{1}, 2);
            PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, 5*nFeatureFunctionsTotal);
            nData = numel(designMatrix);
            PhiCell = repmat(PhiCell, nData, 1);
            
            for n = 1:nData
                %The first columns contain feature function information of the original cell
                PhiCell{n}(:, 1:nFeatureFunctionsTotal) = designMatrix{n};
                
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                for k = 1:obj.coarseScaleDomain.nEl
                    if(mod(k, obj.coarseScaleDomain.nElX) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{n}(k, (nFeatureFunctionsTotal + 1):(2*nFeatureFunctionsTotal)) =...
                           designMatrix{n}(k + 1, :);
                    end
                    
                    if(k <= obj.coarseScaleDomain.nElX*(obj.coarseScaleDomain.nElY - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{n}(k, (2*nFeatureFunctionsTotal + 1):(3*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(k + obj.coarseScaleDomain.nElX, :);
                    end
                    
                    if(mod(k - 1, obj.coarseScaleDomain.nElX) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{n}(k, (3*nFeatureFunctionsTotal + 1):(4*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(k - 1, :);
                    end
                    
                    if(k > obj.coarseScaleDomain.nElX)
                        %lower neighbor of coarse element exists
                        PhiCell{n}(k, (4*nFeatureFunctionsTotal + 1):(5*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(k - obj.coarseScaleDomain.nElX, :);
                    end
                end
            end
            if strcmp(mode, 'train')
                obj.designMatrix = PhiCell;
            elseif strcmp(mode, 'test')
                obj.testDesignMatrix = PhiCell;
            else
                error('wrong mode');
            end
            disp('done')
        end%includeNearestNeighborFeatures
        
        function obj = includeLocalNearestNeighborFeatures(obj, designMatrix, mode)
            %Includes feature function information of neighboring cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest neighbor feature function information separately for each cell...')
            nFeatureFunctionsTotal = size(designMatrix{1}, 2);
            PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, 5*nFeatureFunctionsTotal);
            nData = numel(designMatrix);
            PhiCell = repmat(PhiCell, nData, 1);
            
            for n = 1:nData
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                k = 0;
                for i = 1:obj.coarseScaleDomain.nEl
                    PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                        designMatrix{n}(i, :);
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
                            designMatrix{n}(i + 1, :);
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
                            designMatrix{n}(i + obj.coarseScaleDomain.nElX, :);
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
                            designMatrix{n}(i - 1, :);
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
                            designMatrix{n}(i - obj.coarseScaleDomain.nElX, :);
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
            if strcmp(mode, 'train')
                obj.designMatrix = PhiCell;
            elseif strcmp(mode, 'test')
                obj.testDesignMatrix = PhiCell;
            else
                error('wrong mode');
            end
            disp('done')
        end%includeLocalNearestNeighborFeatures
        
        function obj = includeDiagNeighborFeatures(obj, designMatrix, mode)
            %includes feature function information of all other cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest and diagonal neighbor feature function information...')
            nFeatureFunctionsTotal = size(designMatrix{1}, 2);
            PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, 9*nFeatureFunctionsTotal);
            nData = numel(designMatrix);
            PhiCell = repmat(PhiCell, nData, 1);
            
            for n = 1:nData
                %The first columns contain feature function information of the original cell
                PhiCell{n}(:, 1:nFeatureFunctionsTotal) = designMatrix{n};
                
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                for i = 1:obj.coarseScaleDomain.nEl
                    if(mod(i, obj.coarseScaleDomain.nElX) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{n}(i, (nFeatureFunctionsTotal + 1):(2*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(i + 1, :);
                        if(i <= obj.coarseScaleDomain.nElX*(obj.coarseScaleDomain.nElY - 1))
                            %upper right neighbor of coarse element exists
                            PhiCell{n}(i, (2*nFeatureFunctionsTotal + 1):(3*nFeatureFunctionsTotal)) =...
                                designMatrix{n}(i + obj.coarseScaleDomain.nElX + 1, :);
                        end
                    end
                    
                    if(i <= obj.coarseScaleDomain.nElX*(obj.coarseScaleDomain.nElY - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{n}(i, (3*nFeatureFunctionsTotal + 1):(4*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(i + obj.coarseScaleDomain.nElX, :);
                        if(mod(i - 1, obj.coarseScaleDomain.nElX) ~= 0)
                            %upper left neighbor exists
                            PhiCell{n}(i, (4*nFeatureFunctionsTotal + 1):(5*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(i + obj.coarseScaleDomain.nElX - 1, :);
                        end
                    end
                    
                    if(mod(i - 1, obj.coarseScaleDomain.nElX) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{n}(i, (5*nFeatureFunctionsTotal + 1):(6*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(i - 1, :);
                        if(i > obj.coarseScaleDomain.nElX)
                            %lower left neighbor exists
                            PhiCell{n}(i, (6*nFeatureFunctionsTotal + 1):(7*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(i - obj.coarseScaleDomain.nElX - 1, :);
                        end
                    end
                    
                    if(i > obj.coarseScaleDomain.nElX)
                        %lower neighbor of coarse element exists
                        PhiCell{n}(i, (7*nFeatureFunctionsTotal + 1):(8*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(i - obj.coarseScaleDomain.nElX, :);
                        if(mod(i, obj.coarseScaleDomain.nElX) ~= 0)
                            %lower right neighbor exists
                            PhiCell{n}(i, (8*nFeatureFunctionsTotal + 1):(9*nFeatureFunctionsTotal)) =...
                            designMatrix{n}(i - obj.coarseScaleDomain.nElX + 1, :);
                        end
                    end
                end
            end
            if strcmp(mode, 'train')
                obj.designMatrix = PhiCell;
            elseif strcmp(mode, 'test')
                obj.testDesignMatrix = PhiCell;
            else
                error('wrong mode');
            end
            disp('done')
        end%includeDiagNeighborFeatures

        function obj = includeLocalDiagNeighborFeatures(obj, designMatrix, mode)
            %Includes feature function information of direct and diagonal neighboring cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest + diagonal neighbor feature function information separately for each cell...')
            nFeatureFunctionsTotal = size(designMatrix{1}, 2);
            nData = numel(designMatrix);
%             PhiCell = repmat(PhiCell, nTrain, 1);
            
            for n = 1:nData
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                k = 0;
                for i = 1:obj.coarseScaleDomain.nEl
                    PhiCell{n}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                        designMatrix{n}(i, :);
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
                            designMatrix{n}(i + 1, :);
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
                                designMatrix{n}(i + obj.coarseScaleDomain.nElX + 1, :);
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
                            designMatrix{n}(i + obj.coarseScaleDomain.nElX, :);
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
                                designMatrix{n}(i + obj.coarseScaleDomain.nElX - 1, :);
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
                            designMatrix{n}(i - 1, :);
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
                                designMatrix{n}(i - obj.coarseScaleDomain.nElX - 1, :);
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
                            designMatrix{n}(i - obj.coarseScaleDomain.nElX, :);
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
                                designMatrix{n}(i - obj.coarseScaleDomain.nElX + 1, :);
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
            if strcmp(mode, 'train')
                obj.designMatrix = PhiCell;
            elseif strcmp(mode, 'test')
                obj.testDesignMatrix = PhiCell;
            else
                error('wrong mode');
            end
            disp('done')
        end%includeLocalDiagNeighborFeatures

        function obj = localTheta_c(obj, designMatrix, mode)
            %Sets separate coefficients theta_c for each macro-cell in a single microstructure
            %sample
            %Can never be executed before rescaling/standardization of design Matrix!
            debug = false; %debug mode
            disp('Using separate feature coefficients theta_c for each macro-cell in a microstructure...')
            nFeatureFunctionsTotal = size(designMatrix{1}, 2);
            PhiCell{1} = zeros(obj.coarseScaleDomain.nEl, obj.coarseScaleDomain.nEl*nFeatureFunctionsTotal);
            nData = numel(designMatrix);
            PhiCell = repmat(PhiCell, nData, 1);
            
            %Reassemble design matrix
            for n = 1:nData
                for i = 1:obj.coarseScaleDomain.nEl
                    PhiCell{n}(i, ((i - 1)*nFeatureFunctionsTotal + 1):(i*nFeatureFunctionsTotal)) = ...
                        designMatrix{n}(i, :);
                end
                PhiCell{n} = sparse(PhiCell{n});
            end
            if debug
                firstDesignMatrixBeforeLocal = designMatrix{1}
                firstDesignMatrixAfterLocal = full(PhiCell{1})
                pause
            end
            if strcmp(mode, 'train')
                obj.designMatrix = PhiCell;
            elseif strcmp(mode, 'test')
                obj.testDesignMatrix = PhiCell;
            else
                error('wrong mode');
            end
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
            for i = 1:3
                subplot(1,3,i);
                p(i) = imagesc(reshape(cond(:, samples(i)), obj.fineScaleDomain.nElX, obj.fineScaleDomain.nElY));
                grid off;
                axis square;
                xticks({});
                yticks({});
                if nargin > 2
                    %to plot numerical title
                    title(num2str(titl(i)));
                end
                plotGrid = false;
                if plotGrid
                    for x = 1:(numel(obj.coarseGridVectorX) - 1)
                        line([xLines(x), xLines(x)], [0, obj.nElFX], 'Color', 'w')
                    end
                    for y = 1:(numel(obj.coarseGridVectorY) - 1)
                        line([0, obj.nElFX], [yLines(y), yLines(y)], 'Color', 'w')
                    end
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
            f1 = figure;
            f2 = figure;
            obj = obj.loadTrainingData;
            xLines = cumsum(obj.coarseGridVectorX)*obj.nElFX;
            yLines = cumsum(obj.coarseGridVectorY)*obj.nElFY;
            for i = 1:12
                figure(f1)
                subplot(4, 3, i);
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
                zl = zlabel('$y(\vec{x})$');
                zl.Interpreter = 'latex';
                zl.Rotation = 90;
                zl.FontSize = 26;
                xticklabels({})
                yticklabels({})
                zticklabels({})
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

                figure(f2)
                subplot(4, 3, i);
                [~,p2(i)] = contourf(reshape(Tf(:, samples(i)), (obj.fineScaleDomain.nElX + 1),...
                    (obj.fineScaleDomain.nElY + 1)), 8);
                caxis([min_Tf, max_Tf])
                grid off;
                p2(i).LineStyle = 'none';
                xticks({});
                yticks({});
                axis square;
                colorbar
            end
            
            plotSingleSamples = false;
            if plotSingleSamples
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
            
        end

        function p = plot_p_c_regression(obj, features)
            %Plots regressions of single features to the data <X>_q
            totalFeatures = size(obj.featureFunctions, 2) + size(obj.globalFeatureFunctions, 2);
            if obj.useAutoEnc
                totalFeatures = totalFeatures + obj.latentDim;
            end
            totalFeatures

            iter = 1;
            for feature = features
                if(~(obj.useKernels && totalFeatures ~= 1) || iter == 1)
                    f = figure;
                end
                if obj.useKernels
%                     if size(obj.globalFeatureFunctions, 2)
%                         error('Not yet generalized for global features')
%                     end
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
                                if(size(obj.featureFunctions, 2) + ...
                                       size(obj.globalFeatureFunctions, 2) == 1)
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
                iter = iter + 1;
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
            ylim([(min(obj.thetaArray(end, :)) - 1) (max(obj.thetaArray(end, :)) + 1)]);
            sb = subplot(3,2,2, 'Parent', figHandle);
            bar(obj.theta_c.theta, 'linewidth', 1, 'Parent', sb)
            axis tight;
            sb = subplot(3,2,3, 'Parent', figHandle);
            semilogy(sqrt(obj.sigmaArray), 'linewidth', 1, 'Parent', sb)
            axis tight;
            sb = subplot(3,2,4, 'Parent', figHandle);
            if ~obj.theta_c.full_Sigma
                imagesc(reshape(diag(sqrt(obj.theta_c.Sigma(1:obj.coarseScaleDomain.nEl, 1:obj.coarseScaleDomain.nEl))),...
                    obj.coarseScaleDomain.nElX, obj.coarseScaleDomain.nElY), 'Parent', sb)
            else
                imagesc(reshape(sqrt(diag(obj.theta_c.Sigma)),...
                    obj.coarseScaleDomain.nElX, obj.coarseScaleDomain.nElY), 'Parent', sb)
            end
            title('\sigma_k')
            colorbar
            grid off;
            axis tight;
            sb = subplot(3,2,5, 'Parent', figHandle);
            semilogy(obj.thetaHyperparamArray, 'linewidth', 1, 'Parent', sb)
            axis tight;
            sb = subplot(3,2,6, 'Parent', figHandle);
            bar(obj.thetaPriorHyperparam, 'linewidth', 1, 'Parent', sb)
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
        
        function pcaComponents = globalPCA(obj)
            %Compute PCA on global microstructure - load when file exists
            disp('Performing PCA on global microstructure...')
            if exist(strcat(obj.fineScaleDataPath, 'globalPCA.mat'), 'file')
                load(strcat(obj.fineScaleDataPath, 'globalPCA.mat'));
            else
                if(size(obj.trainingDataMatfile.cond, 2) < obj.pcaSamples)
                    obj.pcaSamples = size(obj.trainingDataMatfile.cond, 2);
                    warning('Less samples than specified are available for PCA')
                end
                
                condUnsup = obj.trainingDataMatfile.cond(:, 1:obj.pcaSamples)';
                pcaComponents = pca(condUnsup, 'NumComponents', obj.globalPcaComponents);
                save(strcat(obj.fineScaleDataPath, 'globalPCA.mat'), 'pcaComponents');
            end
            
            pltPca = false;
            if pltPca
                figure
                for i = 1:min([size(pcaComponents, 2) 25])
                    subplot(5,5,i)
                    imagesc(reshape(pcaComponents(:, i), 256, 256))
                    axis square
                    grid off
                    colorbar
                    xticks({})
                    yticks({})
                end
                drawnow
            end
            disp('done')
        end
        
        function pcaComponents = localPCA(obj)
            %Perform PCA on local macro-cells
            
            disp('Performing PCA on every macro-cell...')
            filename = strcat(obj.fineScaleDataPath, 'localPCA', num2str(obj.coarseGridVectorX),...
                   num2str(obj.coarseGridVectorY), '.mat');
            if exist(filename, 'file')
                load(filename);
            else
                if(size(obj.trainingDataMatfile.cond, 2) < obj.pcaSamples)
                    obj.pcaSamples = size(obj.trainingDataMatfile.cond, 2);
                    warning('Less samples than specified are available for PCA')
                end
                lambdak = obj.get_coarseElementConductivities('train', 1:obj.pcaSamples);
                averageMacroCells = true;
                if averageMacroCells
                    iter = 1;
                    for k = 1:obj.coarseScaleDomain.nEl
                        lambdakArray = zeros(numel(lambdak{1, k}), obj.pcaSamples*obj.coarseScaleDomain.nEl);
                        for n = 1:obj.pcaSamples
                            lambdakArray(:, iter) = lambdak{n, k}(:);
                            iter = iter + 1;
                        end
                    end
                    pcaComponents = pca(lambdakArray', 'NumComponents', obj.localPcaComponents);
                else
                    for k = 1:obj.coarseScaleDomain.nEl
                        lambdakArray = zeros(numel(lambdak{1, k}), obj.pcaSamples);
                        for n = 1:obj.pcaSamples
                            lambdakArray(:, n) = lambdak{n, k}(:);
                        end
                        pcaComponents(:, :, k) = pca(lambdakArray', 'NumComponents', obj.localPcaComponents);
                    end
                end
                save(filename, 'pcaComponents');
            end
            pltPca = false;
            if pltPca
                figure
                for i = 1:min([size(pcaComponents, 2) 25])
                    subplot(5,5,i)
                    imagesc(reshape(pcaComponents(:,i), 64, 64))
                    axis square
                    grid off
                    colorbar
                    xticks({})
                    yticks({})
                end
                drawnow
            end
            disp('done')
        end

        function [obj, phi, phiGlobal, phiConvection, phiGlobalConvection] = setFeatureFunctions(obj)
            %Set up feature function handles;
            %First cell array index is for macro-cell. This allows different features for different
            %macro-cells

            addpath('./featureFunctions')   %Path to feature function library
            conductivities = [obj.lowerConductivity obj.upperConductivity];
            log_cutoff = 1e-5;
            phi = {};
            phiGlobal = {};
            ct = obj.conductivityTransformation;    %avoid broadcasting overhead in designMatrix
            nElf = [obj.nElFX obj.nElFY];
            %constant bias
            for k = 1:obj.coarseScaleDomain.nEl
                nFeatures = 0;
                phi{k, nFeatures + 1} = @(lambda) 1;
                nFeatures = nFeatures + 1;
%                 
                phi{k, nFeatures + 1} = @(lambda)...
                    SCA(lambda, conductivities, ct);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    maxwellGarnett(lambda, conductivities, ct, 'lo');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    maxwellGarnett(lambda, conductivities, ct, 'hi');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    differentialEffectiveMedium(lambda, conductivities, ct, 'lo');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    differentialEffectiveMedium(lambda, conductivities, ct, 'hi');
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda)...
                    log(linealPath(lambda, 4, 'x', 2, conductivities) +...
                    linealPath(lambda, 4, 'y', 2, conductivities) + 1/4096);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    log(linealPath(lambda, 4, 'x', 1, conductivities) +...
                    linealPath(lambda, 4, 'y', 1, conductivities) + 1/4096);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    log(linealPath(lambda, 7, 'x', 2, conductivities) +...
                    linealPath(lambda, 7, 'y', 2, conductivities) + 1/4096);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    log(linealPath(lambda, 7, 'x', 1, conductivities) +...
                    linealPath(lambda, 7, 'y', 1, conductivities) + 1/4096);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    log(linealPath(lambda, 10, 'x', 2, conductivities) +...
                    linealPath(lambda, 10, 'y', 2, conductivities) + 1/4096);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    log(linealPath(lambda, 10, 'x', 1, conductivities) +...
                    linealPath(lambda, 10, 'y', 1, conductivities) + 1/4096);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) linPathParams(lambda, (2:2:8)', conductivities, 1, 'a');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) linPathParams(lambda, (2:2:8)', conductivities, 1, 'b');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) linPathParams(lambda, (2:2:8)', conductivities, 2, 'a');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) linPathParams(lambda, (2:2:8)', conductivities, 2, 'b');
                nFeatures = nFeatures + 1;
%                 phi{k, nFeatures + 1} = @(lambda)...
%                     linealPath(lambda, 3, 'y', 2, conductivities);
%                 nFeatures = nFeatures + 1;
% %                 phi{k, nFeatures + 1} = @(lambda)...
% %                     linealPath(lambda, 3, 'x', 1, conductivities);
% %                 nFeatures = nFeatures + 1;
% %                 phi{k, nFeatures + 1} = @(lambda)...
% %                     linealPath(lambda, 3, 'y', 1, conductivities);
% %                 nFeatures = nFeatures + 1;
% %                 phi{k, nFeatures + 1} = @(lambda)...
% %                     linealPath(lambda, 6, 'x', 1, conductivities);
% %                 nFeatures = nFeatures + 1;
% %                 phi{k, nFeatures + 1} = @(lambda)...
% %                     linealPath(lambda, 6, 'y', 1, conductivities);
% %                 nFeatures = nFeatures + 1;
% % 
                phi{k, nFeatures + 1} = @(lambda)...
                    numberOfObjects(lambda, conductivities, 'hi');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    numberOfObjects(lambda, conductivities, 'lo');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    nPixelCross(lambda, 'y', 1, conductivities, 'max');
				nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    nPixelCross(lambda, 'x', 1, conductivities, 'max');
				nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    nPixelCross(lambda, 'y', 2, conductivities, 'max');
				nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    nPixelCross(lambda, 'x', 2, conductivities, 'max');
				nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'hi', 'y');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'hi', 'x');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'lo', 'y');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'lo', 'x');
                nFeatures = nFeatures + 1;
                

                phi{k, nFeatures + 1} = @(lambda)...
                    conductivityTransform(generalizedMean(lambda, -1), ct);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    conductivityTransform(generalizedMean(lambda, -.5), ct);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    conductivityTransform(generalizedMean(lambda, 0), ct);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    conductivityTransform(generalizedMean(lambda, .5), ct);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    conductivityTransform(generalizedMean(lambda, 1), ct);
                nFeatures = nFeatures + 1;

                phi{k, nFeatures + 1} = @(lambda) log(meanImageProps(lambda,...
                    conductivities, 'hi', 'ConvexArea', 'max') + log_cutoff);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) log(meanImageProps(lambda,...
                    conductivities, 'lo', 'ConvexArea', 'max') + log_cutoff);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) log(meanImageProps(lambda,...
                    conductivities, 'hi', 'ConvexArea', 'var') + log_cutoff);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) log(meanImageProps(lambda,...
                    conductivities, 'lo', 'ConvexArea', 'var') + log_cutoff);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) log(meanImageProps(lambda,...
                    conductivities, 'hi', 'ConvexArea', 'mean') + log_cutoff);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) log(meanImageProps(lambda,...
                    conductivities, 'lo', 'ConvexArea', 'mean') + log_cutoff);
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) ...
                    connectedPathExist(lambda, 1, conductivities, 'x', 'invdist');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) ...
                    connectedPathExist(lambda, 1, conductivities, 'y', 'invdist');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) ...
                    connectedPathExist(lambda, 2, conductivities, 'x', 'invdist');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) ...
                    connectedPathExist(lambda, 2, conductivities, 'y', 'invdist');
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda)...
                    log(specificSurface(lambda, 1, conductivities, nElf) + log_cutoff);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    log(specificSurface(lambda, 2, conductivities, nElf) + log_cutoff);
                nFeatures = nFeatures + 1;

                
                phi{k, nFeatures + 1} = @(lambda)...
                    gaussLinFilt(lambda, nan, 1);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    gaussLinFilt(lambda, nan, 2);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    gaussLinFilt(lambda, nan, 4);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    gaussLinFilt(lambda, nan, 8);
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda)...
                    gaussLinFilt(lambda, nan, 16);
                nFeatures = nFeatures + 1;

                phi{k, nFeatures + 1} = @(lambda) std(lambda(:));
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) log(std(lambda(:)) + log_cutoff);
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) isingEnergy(lambda);
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, -1, 'left');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, -1, 'lower');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, -1, 'right');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, -1, 'upper');
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, 0, 'left');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, 0, 'lower');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, 0, 'right');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, 0, 'upper');
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, 1, 'left');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, 1, 'lower');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, 1, 'right');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) generalizedMeanBoundary(lambda, 1, 'upper');
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'hi',...
                    'euclidean', 'mean');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'hi',...
                    'euclidean', 'var');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'hi',...
                    'euclidean', 'max');
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'hi',...
                    'cityblock', 'mean');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'hi',...
                    'cityblock', 'var');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'hi',...
                    'cityblock', 'max');
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'hi',...
                    'chessboard', 'mean');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'hi',...
                    'chessboard', 'var');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'hi',...
                    'chessboard', 'max');
                nFeatures = nFeatures + 1;

                
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'lo',...
                    'euclidean', 'mean');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'lo',...
                    'euclidean', 'var');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'lo',...
                    'euclidean', 'max');
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'lo',...
                    'cityblock', 'mean');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'lo',...
                    'cityblock', 'var');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'lo',...
                    'cityblock', 'max');
                nFeatures = nFeatures + 1;
                
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'lo',...
                    'chessboard', 'mean');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'lo',...
                    'chessboard', 'var');
                nFeatures = nFeatures + 1;
                phi{k, nFeatures + 1} = @(lambda) distanceProps(lambda, conductivities, 'lo',...
                    'chessboard', 'max');
                nFeatures = nFeatures + 1;

                %Dummy random features
%                 phi{k, nFeatures + 1} = @(lambda) normrnd(0, 1);
%                 nFeatures = nFeatures + 1;
%                 phi{k, nFeatures + 1} = @(lambda) normrnd(0, 1);
%                 nFeatures = nFeatures + 1;
%                 phi{k, nFeatures + 1} = @(lambda) normrnd(0, 1);
%                 nFeatures = nFeatures + 1;
            end
            
            %Global features
            for k = 1:obj.coarseScaleDomain.nEl
                nGlobalFeatures = 0;
%                 phiGlobal{k, nGlobalFeatures + 1} = @(lambda) ...
%                     connectedPathExist(lambda, 2, conductivities, 'x', 'invdist');
%                 nGlobalFeatures = nGlobalFeatures + 1;
%                 phiGlobal{k, nGlobalFeatures + 1} = @(lambda) ...
%                     connectedPathExist(lambda, 2, conductivities, 'y', 'invdist');
%                 nGlobalFeatures = nGlobalFeatures + 1;
                phiGlobal{k, nGlobalFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'hi', 'x');
                nGlobalFeatures = nGlobalFeatures + 1;
                phiGlobal{k, nGlobalFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'hi', 'y');
                nGlobalFeatures = nGlobalFeatures + 1;
                phiGlobal{k, nGlobalFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'lo', 'x');
                nGlobalFeatures = nGlobalFeatures + 1;
                phiGlobal{k, nGlobalFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'lo', 'y');
                nGlobalFeatures = nGlobalFeatures + 1;
                phiGlobal{k, nGlobalFeatures + 1} = @(lambda)...
                    SCA(lambda, conductivities, ct);
                nGlobalFeatures = nGlobalFeatures + 1;
                phiGlobal{k, nGlobalFeatures + 1} = @(lambda)...
                    maxwellGarnett(lambda, conductivities, ct, 'lo');
                nGlobalFeatures = nGlobalFeatures + 1;
                phiGlobal{k, nGlobalFeatures + 1} = @(lambda)...
                    maxwellGarnett(lambda, conductivities, ct, 'hi');
                nGlobalFeatures = nGlobalFeatures + 1;
                phiGlobal{k, nGlobalFeatures + 1} = @(lambda)...
                    differentialEffectiveMedium(lambda, conductivities, ct, 'lo');
                nGlobalFeatures = nGlobalFeatures + 1;
                phiGlobal{k, nGlobalFeatures + 1} = @(lambda)...
                    differentialEffectiveMedium(lambda, conductivities, ct, 'hi');
                nGlobalFeatures = nGlobalFeatures + 1;
            end
            
            pltPca = false;
            %Unsupervised pretraining: compute PCA components
            if(obj.globalPcaComponents > 0)
                globalComponents = obj.globalPCA;
            end
            %PCA projection
            for n = 1:obj.globalPcaComponents
                for k = 1:obj.coarseScaleDomain.nEl
                    phiGlobal{k, nGlobalFeatures + n} = @(lambda) globalComponents(:, n)'*lambda(:);
                end
            end
            
            %local PCA projection
            if(obj.localPcaComponents > 0)
                localComponents = obj.localPCA;
                for n = 1:obj.localPcaComponents
                    for k = 1:obj.coarseScaleDomain.nEl
                        if(ndims(localComponents) == 3)
                            %Separate PCA on every macro cell
                            phi{k, nFeatures + n} = @(lambda) localComponents(:, n, k)'*lambda(:);
                        else
                            phi{k, nFeatures + n} = @(lambda) localComponents(:, n)'*lambda(:);
                        end
                    end
                end
            end
            
            obj.secondOrderTerms = zeros(nFeatures, 'logical');
%             obj.secondOrderTerms(2, 2) = true;
%             obj.secondOrderTerms(2, 3) = true;
%             obj.secondOrderTerms(3, 3) = true;
            assert(sum(sum(tril(obj.secondOrderTerms, -1))) == 0, 'Second order matrix must be upper triangular')
            
            
            %Convection features
            if obj.useConvection
                for k = 1:obj.coarseScaleDomain.nEl
                    phiConvection{k, 1} = @(convField) mean(mean(convField(1, :, :)));
                    phiConvection{k, 2} = @(convField) mean(mean(convField(2, :, :)));
                end
            else
                phiConvection = {};
            end
            
            obj.featureFunctions = phi;
            obj.globalFeatureFunctions = phiGlobal;
            obj.convectionFeatureFunctions = phiConvection;
            phiGlobalConvection = {};
            obj.globalConvectionFeatureFunctions = phiGlobalConvection;
            
            %add previously learned linear filters
            if(~isempty(obj.linFilt))
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
            end
        end

        function obj = setCoarseGrid(obj, coarseGridX, coarseGridY)
            %coarseGridX and coarseGridY are coarse model grid vectors
            obj.coarseGridVectorX = coarseGridX;
            obj.coarseGridVectorY = coarseGridY;
        end
    end
    
end


















