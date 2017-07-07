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
                
        %% Model training parameters
        nStart = 1;             %first training data sample in file
        nTrain = 16;            %number of samples used for training
        mode = 'useLocal';      %useNeighbor, useLocalNeighbor, useDiagNeighbor, useLocalDiagNeighbor, useLocal, global
                                %global: take whole microstructure as feature function input, not
                                %only local window (only recommended for pooling)
        linFiltSeq = false;
        
        %% Model parameters
        theta_c;
        theta_cf;
        featureFunctions;       %Cell array containing local feature function handles
        globalFeatureFunctions  %cell array with handles to global feature functions
        %transformation of finescale conductivity to real axis
        conductivityTransformation;
        
        %% Feature function rescaling parameters
        featureFunctionMean;
        featureFunctionSqMean;
        featureFunctionMin;
        featureFunctionMax;
        standardizeFeatures = false;    %Rescale s.t. feature outputs have mean 0 and std 1
        rescaleFeatures = false;        %Rescale s.t. feature outputs are all between 0 and 1
        
        %% Prediction parameters
        nSamples_p_c = 1000;
        testSamples;       %pick out specific test samples here
        
        %% Prediction outputs
        predMeanArray;
        predVarArray;
        meanPredMeanOutput;                %mean predicted mean of output field
        meanSquaredDistance;               %mean squared distance of predicted mean to true solution
        meanSquaredDistanceError;          %Monte Carlo error
        meanMahalanobisError;
    end
    
    
    
    
    properties(SetAccess = private)
        %% finescale data specifications
        conductivityLengthScaleDist = 'lognormal';      %delta for fixed length scale, lognormal for rand
        conductivityDistributionParams = {0.2 [.08 .08] 1};     %for correlated_binary: 
                                                                %{volumeFraction, correlationLength, sigma_f2}
                                                                %for log normal length scale, the
                                                                %length scale parameters are log normal mu and
                                                                %sigma
        %Coefficients giving boundary conditions, specify as string
        boundaryConditions = '[0 1000 0 0]';
        
        %% Coarse model specifications
        coarseScaleDomain;
        coarseGridVectorX = [1/2 1/2];
        coarseGridVectorY = [1/2 1/2];
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
                        
            %Set up default value for test samples
            obj.testSamples = 1:obj.nSets(2);
            
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
                    corrLength2 = obj.conductivityDistributionParams{2}(1);
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
            if obj.linFiltSeq
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
        
        
        
        
        function obj = predict(obj)
            %Function to predict finescale output from generative model
            
            %Load test file
            Tf = obj.testDataMatfile.Tf(:, obj.testSamples);
            obj = obj.loadTrainedParams;

            %to find DesignMatrix class
            addpath('./rom')
            %% Compute design matrices
            Phi = DesignMatrix(obj.fineScaleDomain, obj.coarseScaleDomain, obj.featureFunctions,...
                obj.globalFeatureFunctions, obj.testDataMatfile, obj.testSamples);
            addpath('./aux')    %for conductivityBackTransform
            Phi = Phi.computeDesignMatrix(obj.coarseScaleDomain.nEl, obj.fineScaleDomain.nEl,...
                obj.conductivityTransformation);
            %Normalize design matrices
            if obj.rescaleFeatures
                Phi = Phi.rescaleDesignMatrix(obj.featureFunctionMin, obj.featureFunctionMax);
            elseif obj.standardizeFeatures
                Phi = Phi.standardizeDesignMatrix(obj.featureFunctionMean, obj.featureFunctionSqMean);
            end
            if strcmp(obj.mode, 'useNeighbor')
                %use feature function information from nearest neighbors
                Phi = Phi.includeNearestNeighborFeatures([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            elseif strcmp(obj.mode, 'useLocalNeighbor')
                Phi = Phi.includeLocalNearestNeighborFeatures([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            elseif strcmp(obj.mode, 'useLocalDiagNeighbor')
                Phi = Phi.includeLocalDiagNeighborFeatures([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            elseif strcmp(obj.mode, 'useDiagNeighbor')
                %use feature function information from nearest and diagonal neighbortras
                Phi = Phi.includeDiagNeighborFeatures([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            elseif strcmp(obj.mode, 'useLocal')
                Phi = Phi.localTheta_c([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            end

            %% Sample from p_c
            disp('Sampling from p_c...')
            nTest = numel(obj.testSamples);
            Xsamples = zeros(obj.coarseScaleDomain.nEl, obj.nSamples_p_c, nTest);
            LambdaSamples{1} = zeros(obj.coarseScaleDomain.nEl, obj.nSamples_p_c);
            LambdaSamples = repmat(LambdaSamples, nTest, 1);
            meanEffCond = zeros(obj.coarseScaleDomain.nEl, nTest);
            
            useNeuralNet = false;
            if useNeuralNet
                %Only valid for square grids!!!
                finePerCoarse = [sqrt(size(Phi.xk{1}, 1)), sqrt(size(Phi.xk{1}, 1))];
                xkNN = zeros(finePerCoarse(1), finePerCoarse(2), 1, nTest*obj.coarseScaleDomain.nEl);
                k = 1;
                for i = 1:nTest
                    for j = 1:obj.coarseScaleDomain.nEl
                        xkNN(:, :, 1, k) =...
                            reshape(Phi.xk{i}(:, j), finePerCoarse(1), finePerCoarse(2)); %for neural net
                        k = k + 1;
                    end
                end
            end
            
            for i = 1:nTest
                if useNeuralNet
                    PhiMat = xkNN(:, :, 1, ((i - 1)*obj.coarseScaleDomain.nEl + 1):(i*obj.coarseScaleDomain.nEl));
                    mu = double(predict(obj.theta_c.theta, PhiMat));
                    Xsamples(:, :, i) = mvnrnd(mu, obj.theta_c.Sigma, obj.nSamples_p_c)';
                else
                    Xsamples(:, :, i) = mvnrnd(Phi.designMatrices{i}*obj.theta_c.theta,...
                        obj.theta_c.Sigma, obj.nSamples_p_c)';
                end
                LambdaSamples{i} = conductivityBackTransform(Xsamples(:, :, i), obj.conductivityTransformation);
                if(strcmp(obj.conductivityTransformation.type, 'log') && ~useNeuralNet)
                    meanEffCond(:, i) = exp(Phi.designMatrices{i}*obj.theta_c.theta + .5*diag(obj.theta_c.Sigma));
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
                cond = obj.testDataMatfile.cond(:, pstart:(pstart + 5));
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
        
        
        
        
        function p = plot_p_c_regression(obj, Phi, XMean)
            %Plots regressions of single features to the data <X>_q
            totalFeatures = size(obj.featureFunctions, 2) + size(obj.globalFeatureFunctions, 2);
            for feature = 1:min([4, totalFeatures])
                k = 1;
                f = figure;
                mink = Inf*ones(obj.coarseScaleDomain.nEl, 1);
                maxk = -Inf*ones(obj.coarseScaleDomain.nEl, 1);
                if strcmp(obj.mode, 'useLocal')
                    for i = 1:obj.coarseScaleDomain.nElX
                        for j = 1:obj.coarseScaleDomain.nElY
                            subplot(obj.coarseScaleDomain.nElX, obj.coarseScaleDomain.nElY, k);
                            for s = 1:obj.nTrain
                                yData(k, s) = XMean(k, s);
                                for l = 1:totalFeatures
                                    if l~= feature
                                    yData(k, s) = yData(k, s) -...
                                        obj.theta_c.theta(totalFeatures*(k - 1) + l)*...
                                        Phi.designMatrices{s}(k, totalFeatures*(k - 1) + l);
                                    end
                                end
                                plot(Phi.designMatrices{s}(k, totalFeatures*(k - 1) + feature), yData(k, s), 'xb')
                                if(Phi.designMatrices{s}(k, totalFeatures*(k - 1) + feature) < mink(k))
                                    mink(k) = Phi.designMatrices{s}(k, totalFeatures*(k - 1) + feature);
                                elseif(Phi.designMatrices{s}(k, totalFeatures*(k - 1) + feature) > maxk(k))
                                    maxk(k) = Phi.designMatrices{s}(k, totalFeatures*(k - 1) + feature);
                                end
                                hold on;
                            end
                            x = linspace(mink(k), maxk(k), 10);
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
                elseif strcmp(romObj.mode, 'none')
                    for i = 1:romObj.coarseScaleDomain.nElX
                        for j = 1:romObj.coarseScaleDomain.nElY
                            for s = 1:romObj.nTrain
                                plot(Phi.designMatrices{s}(k, feature), XMean(k, s), 'xb')
                                hold on;
                            end
                            x = linspace(min(min(cell2mat(Phi.designMatrices))),...
                                max(max(cell2mat(Phi.designMatrices))), 100);
                            y = romObj.theta_c.theta(feature)*x;
                            plot(x, y);
                            axis tight;
                            k = k + 1;
                        end
                    end
                end
            end
        end
        
        
        
        
        function [obj, Phi] = addLinearFilterFeature(obj, XMean, Phi)
            %Phi is a design matrix object
            
            assert(strcmp(obj.mode, 'useLocal'),...
                'Error: sequential addition of linear filters only working in useLocal mode');
            
%             sigma2Inv_vec = (1./diag(obj.theta_c.Sigma));
            XMeanMinusPhiThetac = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
            for i = 1:obj.nTrain
                XMeanMinusPhiThetac(:, i) = XMean(:, i) - Phi.designMatrices{i}*obj.theta_c.theta;
            end
            
            %We use different linear filters for different macro-cells k
            w{1} = 0;
            w = repmat(w, obj.coarseScaleDomain.nEl, 1);
            E = zeros(1, obj.coarseScaleDomain.nEl);
            for m = 1:obj.coarseScaleDomain.nEl
                for i = 1:obj.nTrain
%                     w{m} = w{m} + sigma2Inv_vec(m)*XMeanMinusPhiThetac(m, i)*Phi.xk{i, m}(:);
                    %should be still correct without the sigma and subsequent normalization
                    w{m} = w{m} + XMeanMinusPhiThetac(m, i)*Phi.xk{i, m}(:);
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
                Phi.featureFunctions{m, nFeaturesBefore + 1} = obj.featureFunctions{m, nFeaturesBefore + 1};
            end
            
            save(filename, 'w_all');
            %save E
            filename = './data/E';
            save(filename, 'E', '-ascii', '-append');
            
            f = figure;
            for m = 1:obj.coarseScaleDomain.nEl
                subplot(obj.coarseScaleDomain.nElX, obj.coarseScaleDomain.nElY, m);
                imagesc(reshape(w{m}, size(Phi.xk{1, m})))
                axis square
                grid off
                xticks({})
                yticks({})
                colorbar
            end
            drawnow
            
            %% recompute design matrices
            %this can be done more efficiently!
%             Phi = Phi.addLinearFilter(w);
            Phi = Phi.computeDesignMatrix(obj.coarseScaleDomain.nEl, obj.fineScaleDomain.nEl,...
                obj.conductivityTransformation);
            Phi = Phi.localTheta_c([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            %Compute sum_i Phi^T(x_i)^Phi(x_i)
            Phi = Phi.computeSumPhiTPhi;
            Phi.sumPhiTPhi = sparse(Phi.sumPhiTPhi);
            
            %% append theta-value
            nTotalFeaturesAfter = size(Phi.designMatrices{1}, 2);
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
        
        
        
        
        
        function [obj, Phi] = addGlobalLinearFilterFeature(obj, XMean, Phi)
            %Phi is a design matrix object
            
            assert(strcmp(obj.mode, 'useLocal'),...
                'Error: sequential addition of linear filters only working in useLocal mode');
            XMeanMinusPhiThetac = zeros(obj.coarseScaleDomain.nEl, obj.nTrain);
            for i = 1:obj.nTrain
                XMeanMinusPhiThetac(:, i) = XMean(:, i) - Phi.designMatrices{i}*obj.theta_c.theta;
            end
            
            %We use different linear filters for different macro-cells k
            w{1} = 0;
            w = repmat(w, obj.coarseScaleDomain.nEl, 1);
            EGlobal = zeros(1, obj.coarseScaleDomain.nEl);
            for m = 1:obj.coarseScaleDomain.nEl
                for i = 1:obj.nTrain
                    w{m} = w{m} + XMeanMinusPhiThetac(m, i)*Phi.transformedConductivity{i}(:);
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
                Phi.globalFeatureFunctions{m, nGlobalFeaturesBefore + 1} =...
                    obj.globalFeatureFunctions{m, nGlobalFeaturesBefore + 1};
            end
            
            save(filename, 'w_allGlobal');
            %save E
            filename = './data/EGlobal';
            save(filename, 'EGlobal', '-ascii', '-append');
            
            f = figure;
            for m = 1:obj.coarseScaleDomain.nEl
                subplot(obj.coarseScaleDomain.nElX, obj.coarseScaleDomain.nElY, m);
                imagesc(reshape(w{m}, size(Phi.transformedConductivity{1})))
                axis square
                grid off
                xticks({})
                yticks({})
                colorbar
            end
            drawnow
            
            %% recompute design matrices
            %this can be done more efficiently!
            %             Phi = Phi.addLinearFilter(w);
            Phi = Phi.computeDesignMatrix(obj.coarseScaleDomain.nEl, obj.fineScaleDomain.nEl,...
                obj.conductivityTransformation);
            Phi = Phi.localTheta_c([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            %Compute sum_i Phi^T(x_i)^Phi(x_i)
            Phi = Phi.computeSumPhiTPhi;
            Phi.sumPhiTPhi = sparse(Phi.sumPhiTPhi);
            
            %% extend theta vector
            nTotalFeaturesAfter = size(Phi.designMatrices{1}, 2);
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
            Phi = DesignMatrix(obj.fineScaleDomain, obj.coarseScaleDomain, obj.featureFunctions,...
                obj.globalFeatureFunctions, obj.trainingDataMatfile, obj.nStart:(obj.nStart + obj.nTrain - 1));
            Phi = Phi.computeDesignMatrix(obj.coarseScaleDomain.nEl, obj.fineScaleDomain.nEl,...
                obj.conductivityTransformation);
            %Normalize design matrices
            if obj.rescaleFeatures
                Phi = Phi.rescaleDesignMatrix(obj.featureFunctionMin, obj.featureFunctionMax);
            elseif obj.standardizeFeatures
                Phi = Phi.standardizeDesignMatrix(obj.featureFunctionMean, obj.featureFunctionSqMean);
            end
            if strcmp(obj.mode, 'useNeighbor')
                %use feature function information from nearest neighbors
                Phi = Phi.includeNearestNeighborFeatures([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            elseif strcmp(obj.mode, 'useLocalNeighbor')
                Phi = Phi.includeLocalNearestNeighborFeatures([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            elseif strcmp(obj.mode, 'useLocalDiagNeighbor')
                Phi = Phi.includeLocalDiagNeighborFeatures([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            elseif strcmp(obj.mode, 'useDiagNeighbor')
                %use feature function information from nearest and diagonal neighbors
                Phi = Phi.includeDiagNeighborFeatures([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
            elseif strcmp(obj.mode, 'useLocal')
                Phi = Phi.localTheta_c([obj.coarseScaleDomain.nElX obj.coarseScaleDomain.nElY]);
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
                mu_i = Phi.designMatrices{i}*obj.theta_c.theta;
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
                obj.featureFunctions{k, nFeatures + 1} = @(lambda) 1;
                nFeatures = nFeatures + 1;
                
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    SCA(lambda, conductivities, obj.conductivityTransformation);
                nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    maxwellGarnett(lambda, conductivities, obj.conductivityTransformation, 'lo');
                nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    differentialEffectiveMedium(lambda, conductivities, obj.conductivityTransformation, 'lo');
                nFeatures = nFeatures + 1;
                
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    linealPath(lambda, 3, 'x', 2, conductivities);
                nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    linealPath(lambda, 3, 'y', 2, conductivities);
                nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    linealPath(lambda, 3, 'x', 1, conductivities);
                nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    linealPath(lambda, 3, 'y', 1, conductivities);
                nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    linealPath(lambda, 6, 'x', 1, conductivities);
                nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    linealPath(lambda, 6, 'y', 1, conductivities);
                nFeatures = nFeatures + 1;

                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    numberOfObjects(lambda, conductivities, 'hi');
                nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    nPixelCross(lambda, 'y', 1, conductivities, 'mean');
				nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    nPixelCross(lambda, 'x', 1, conductivities, 'mean');
				nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'hi', 'y');
                nFeatures = nFeatures + 1;
                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    maxExtent(lambda, conductivities, 'hi', 'x');
                nFeatures = nFeatures + 1;
                

                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    conductivityTransform(generalizedMean(lambda, -1), obj.conductivityTransformation);
                nFeatures = nFeatures + 1;

                obj.featureFunctions{k, nFeatures + 1} = @(lambda) log(meanImageProps(lambda,...
                    conductivities, 'hi', 'ConvexArea', 'max') + log_cutoff);
                nFeatures = nFeatures + 1;
                
%                 obj.featureFunctions{k, nFeatures + 1} = @(lambda) ...
%                     connectedPathExist(lambda, 2, conductivities, 'x', 'invdist');
%                 nFeatures = nFeatures + 1;
%                 obj.featureFunctions{k, nFeatures + 1} = @(lambda) ...
%                     connectedPathExist(lambda, 2, conductivities, 'y', 'invdist');
%                 nFeatures = nFeatures + 1;
                
%                 obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
%                     log(specificSurface(lambda, 2, conductivities, [obj.nElFX obj.nElFY]) + log_cutoff);
%                 nFeatures = nFeatures + 1;

                obj.featureFunctions{k, nFeatures + 1} = @(lambda)...
                    gaussLinFilt(lambda);
                nFeatures = nFeatures + 1;
            end
            
        end
        
        
        
        
        function obj = setCoarseGrid(obj, coarseGridX, coarseGridY)
            %coarseGridX and coarseGridY are coarse model grid vectors
            obj.coarseGridVectorX = coarseGridX;
            obj.coarseGridVectorY = coarseGridY;
        end
    end
    
end


















