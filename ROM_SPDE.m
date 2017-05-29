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
        nTrain = 32;            %number of samples used for training
        mode = 'useLocal';          %useNeighbor, useLocalNeighbor, useDiagNeighbor, useLocalDiagNeighbor, useLocal, global
                                %global: take whole microstructure as feature function input, not
                                %only local window (only recommended for pooling)
        linFiltSeq = true;
        
        %% Model parameters
        theta_c;
        theta_cf;
        featureFunctions;       %Cell array containing feature function handles
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
        conductivityDistributionParams = {.2 [.08 .08] 1};      %for correlated_binary: 
                                                            %{volumeFraction, correlationLength, sigma_f2}
        %Coefficients giving boundary conditions, specify as string
        boundaryConditions = '[0 1000 0 0]';
        
        %% Coarse model specifications
        coarseScaleDomain;
        coarseGridVectorX = [.25 .25 .25 .25];
        coarseGridVectorY = [.25 .25 .25 .25];
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
                obj.conductivityTransformation.limits = [.9*obj.lowerConductivity 1.1*obj.upperConductivity];
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
                    p{i} = genBochnerSamples(obj.conductivityDistributionParams{2}(1),...
                        obj.conductivityDistributionParams{3}, nBochnerBasis);
                end
                nEl = obj.fineScaleDomain.nEl;
                upCond = obj.upperConductivity;
                loCond = obj.lowerConductivity;
                cutoff = norminv(1 - obj.conductivityDistributionParams{1}, 0, obj.conductivityDistributionParams{3});
                parfor i = 1:(obj.nSets(nSet))
                    %use for-loop instead of vectorization to save memory
                    for j = 1:nEl
                        ps = p{i}(x(:, j));
                        cond{i}(j) = upCond*(ps > cutoff) +...
                            loCond*(ps <= cutoff);
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
            corrLength = obj.conductivityDistributionParams{2}(1);  %isotropic length scale distribution
            sigma_f2 = obj.conductivityDistributionParams{3};
            obj.fineScaleDataPath = strcat(obj.fineScaleDataPath,...
                'systemSize=', num2str(obj.nElFX), 'x', num2str(obj.nElFY), '/');
            %Type of conductivity distribution
            if strcmp(obj.conductivityDistribution, 'correlated_binary')
                obj.fineScaleDataPath = strcat(obj.fineScaleDataPath,...
                    obj.conductivityDistribution, '/', 'IsoSEcov/', 'l=',...
                    num2str(corrLength), '_sigmafSq=', num2str(sigma_f2),...
                    '/volumeFraction=', num2str(volFrac), '/', 'locond=',...
                    num2str(obj.lowerConductivity), '_upcond=', num2str(obj.upperConductivity),...
                    '/', 'BCcoeffs=', obj.boundaryConditions, '/');
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
            obj = obj.setFeatureFunctions;  %is there a better solution than executing a script?
            if obj.linFiltSeq
                w = dlmread('./data/w');
                for i = 1:size(w, 1)
                    obj.featureFunctions{end + 1} = @(lambda) sum(w(i, :)'.*log(lambda(:)));
                end
            end
            disp('done')
%             rmpath('./rom')
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
                obj.testDataMatfile, obj.testSamples);
            load('./data/conductivityTransformation.mat');
            Phi = Phi.computeDesignMatrix(obj.coarseScaleDomain.nEl, obj.fineScaleDomain.nEl,...
                obj.conductivityTransformation, obj.mode);
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
            
            addpath('./aux')    %for conductivityBackTransform
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
            t_c = obj.theta_c;
            
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
                    TfMeanArray{j} = ((i - 1)/i)*TfMeanArray{j} + (1/i)*mu_cf;  %U_f-integration can be done analytically
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
%             rmpath('./rom')
%             rmpath('./heatFEM')
            
        end
        
        
        
        
        %% plot functions
        function [p] = plotTrainingInput(obj, samples)
            %Load microstructures
            cond = obj.trainingDataMatfile.cond(:, samples);
            f = figure;
            obj = obj.loadTrainingData;
            for i = 1:6
                subplot(2,3,i);
                p(i) = imagesc(reshape(cond(:, samples(i)), obj.fineScaleDomain.nElX, obj.fineScaleDomain.nElY));
                grid off;
                axis square;
                xticks({});
                yticks({});
            end
            
        end
        
        
        
        
        function obj = addLinearFilterFeature(obj)
            
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
            %Set up feature function handles

            addpath('./featureFunctions')   %Path to feature function library
            conductivities = [obj.lowerConductivity obj.upperConductivity];
            obj.featureFunctions = {};
            %constant bias
            obj.featureFunctions{end + 1} = @(lambda) 1;
            %Maxwell-Garnett approximation
%             obj.featureFunctions{end + 1} = @(lambda) maxwellGarnett(lambda, conductivities, 'log', 'hi');
            %SCA
%             obj.featureFunctions{end + 1} = @(lambda) SCA(lambda, conductivities, obj.conductivityTransformation);
%             obj.featureFunctions{end + 1} = @(lambda) generalizedMean(lambda, 0);
        end
        
        
        
        
        function obj = setCoarseGrid(obj, coarseGridX, coarseGridY)
            %coarseGridX and coarseGridY are coarse model grid vectors
            obj.coarseGridVectorX = coarseGridX;
            obj.coarseGridVectorY = coarseGridY;
        end
    end
    
end


















