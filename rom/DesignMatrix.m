classdef DesignMatrix
    %Class describing the design matrices Phi for different data points
    
    properties
        
        designMatrices          %Design matrices stored in cells
        
        dataFile                %mat file holding the training/test data
        dataSamples             %vector with data sample indices
        
        featureFunctions        %Cell array of handles to feature functions
        featureFunctionMean  %mean absolute output of feature function over training set BEFORE normalization
        featureFunctionSqMean
        featureFunctionStd
        featureFunctionMin
        featureFunctionMax
        
        E                       %gives the coarse element a fine element belongs to
        sumPhiTPhi
        
    end
    
    methods
        
        %constructor
        function Phi = DesignMatrix(nf, nc, featureFunctions, dataFile, dataSamples)
            %Set up mapping from fine to coarse element
            Phi = getCoarseElement(Phi, nf, nc);
            Phi.featureFunctions = featureFunctions;
            Phi.dataFile = dataFile;
            Phi.dataSamples = dataSamples;
            
        end
        
        function Phi = getCoarseElement(Phi, nf, nc)
            %Takes element number of full order model, gives element number of
            %coarse model
            %nf, nc are 2D vectors holding the element numbers in x- and y-direction
            
            fineElements = 1:(prod(nf));
            
            %fine elements per coarse mesh
            fine_per_coarse = nf./nc;
            %must be integer
            assert(~any(mod(nf, nc)), 'Error: no integer number of fine elements within a coarse element')
            
            row_fine = floor((fineElements - 1)/nf(1) + 1);
            col_fine = mod((fineElements - 1), nf(1)) + 1;
            
            row_coarse = floor((row_fine - 1)/fine_per_coarse(2) + 1);
            col_coarse = floor((col_fine - 1)/fine_per_coarse(1) + 1);
            
            Phi.E = (row_coarse - 1)*nc(1) + col_coarse;
            
        end
        
        function Phi = computeDesignMatrix(Phi, nElc, nElf, condTransOpts)
            %Actual computation of design matrix
            tic
            disp('Compute design matrices Phi...')
            
            %load finescale conductivity field
            conductivity = Phi.dataFile.cond(:, Phi.dataSamples);
            conductivity = num2cell(conductivity, 1);   %to avoid parallelization communication overhead
            nTrain = length(Phi.dataSamples);
            nFeatureFunctions = numel(Phi.featureFunctions);
            phi = Phi.featureFunctions;
            coarseElement = Phi.E;
            
            %Open parallel pool
            addpath('./computation')
            parPoolInit(nTrain);
            if condTransOpts.anisotropy
                PhiCell{1} = zeros(3*nElc, nFeatureFunctions);
            else
                PhiCell{1} = zeros(nElc, nFeatureFunctions);
            end
            PhiCell = repmat(PhiCell, nTrain, 1);
            parfor s = 1:nTrain
                %inputs belonging to same coarse element are in the same column of xk. They are ordered in
                %x-direction.
                if condTransOpts.anisotropy
                    PhiCell{s} = zeros(3*nElc, nFeatureFunctions);
                else
                    PhiCell{s} = zeros(nElc, nFeatureFunctions);
                end
                lambdak = zeros(nElf/nElc, nElc);
                for i = 1:nElc
                    lambdak(:, i) = conductivity{s}(coarseElement == i);
                end
                
                %construct design matrix Phi
                for i = 1:nElc
                    for j = 1:nFeatureFunctions
                        if condTransOpts.anisotropy
                            PhiCell{s}((1 + (i - 1)*3):(i*3), j) = phi{j}(lambdak(:, i));
                        else
                            PhiCell{s}(i, j) = phi{j}(lambdak(:, i));                           
                        end
                    end
                end
            end
            
            Phi.designMatrices = PhiCell;
            %Check for real finite inputs
            for i = 1:nTrain
                if(~all(all(all(isfinite(Phi.designMatrices{i})))))
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(Phi.designMatrices{i}),...
                        find(~isfinite(Phi.designMatrices{i})))
                    warning('Non-finite design matrix Phi. Setting non-finite component to 0.')
                    Phi.designMatrices{i}(~isfinite(Phi.designMatrices{i})) = 0;
                elseif(~all(all(all(isreal(Phi.designMatrices{i})))))
                    warning('Complex feature function output:')
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(Phi.designMatrices{i}),...
                        find(imag(Phi.designMatrices{i})))
                    disp('Ignoring imaginary part...')
                    Phi.designMatrices{i} = real(Phi.designMatrices{i});
                end
            end
            disp('done')
            Phi_computation_time = toc
            
        end
        
        function Phi = includeNearestNeighborFeatures(Phi, nc)
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest neighbor feature function information...')
            nElc = prod(nc);
            nFeatureFunctions = numel(Phi.featureFunctions);
            PhiCell{1} = zeros(nElc, 5*nFeatureFunctions);
            nTrain = length(Phi.dataSamples);
            PhiCell = repmat(PhiCell, nTrain, 1);
            
            for s = 1:nTrain
                PhiCell{s}(:, 1:nFeatureFunctions) = Phi.designMatrices{s};
                
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                for i = 1:nElc
                    if(mod(i, nc(1)) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{s}(i, (nFeatureFunctions + 1):(2*nFeatureFunctions)) =...
                           Phi.designMatrices{s}(i + 1, :);
                    end
                    
                    if(i < nc(1)*(nc(2) - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{s}(i, (2*nFeatureFunctions + 1):(3*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i + nc(1), :);
                    end
                    
                    if(mod(i - 1, nc(1)) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{s}(i, (3*nFeatureFunctions + 1):(4*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i - 1, :);
                    end
                    
                    if(i > nc(1))
                        %lower neighbor of coarse element exists
                        PhiCell{s}(i, (4*nFeatureFunctions + 1):(5*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i - nc(1), :);
                    end
                end
            end
            Phi.designMatrices = PhiCell;
            
        end%includeNearestNeighborFeatures
        
        function Phi = computeFeatureFunctionMinMax(Phi)
            %Computes min/max of feature function outputs over training data
            Phi.featureFunctionMin = min(Phi.designMatrices{1});
            Phi.featureFunctionMax = max(Phi.designMatrices{1});
            for i = 1:numel(Phi.designMatrices)
                min_i = min(Phi.designMatrices{i});
                max_i = max(Phi.designMatrices{i});
                Phi.featureFunctionMin(Phi.featureFunctionMin > min_i) = min_i(Phi.featureFunctionMin > min_i);
                Phi.featureFunctionMax(Phi.featureFunctionMax < max_i) = max_i(Phi.featureFunctionMax < max_i);
            end
        end
        
        function Phi = computeFeatureFunctionMean(Phi)
            Phi.featureFunctionMean = 0;
            for i = 1:numel(Phi.designMatrices)
                %                 Phi.featureFunctionMean = Phi.featureFunctionMean + mean(abs(Phi.designMatrices{i}), 1);
                Phi.featureFunctionMean = Phi.featureFunctionMean + mean(Phi.designMatrices{i}, 1);
            end
            Phi.featureFunctionMean = Phi.featureFunctionMean/numel(Phi.designMatrices);
        end
        
        function Phi = computeFeatureFunctionSqMean(Phi)
            featureFunctionSqSum = 0;
            for i = 1:numel(Phi.designMatrices)
                featureFunctionSqSum = featureFunctionSqSum + sum(Phi.designMatrices{i}.^2, 1);
            end
            Phi.featureFunctionSqMean = featureFunctionSqSum/...
                (numel(Phi.designMatrices)*size(Phi.designMatrices{1}, 1));
        end
        
        function Phi = rescaleDesignMatrix(Phi, featFuncMin, featFuncMax)
            %Rescale design matrix s.t. outputs are between 0 and 1
            disp('Rescale design matrix...')
            if(nargin > 1)
                for i = 1:numel(Phi.designMatrices)
                    Phi.designMatrices{i} = (Phi.designMatrices{i} - featFuncMin)./...
                        (featFuncMax - featFuncMin);
                end
            else
                Phi = Phi.computeFeatureFunctionMinMax;
                for i = 1:numel(Phi.designMatrices)
                    Phi.designMatrices{i} = (Phi.designMatrices{i} - Phi.featureFunctionMin)./...
                        (Phi.featureFunctionMax - Phi.featureFunctionMin);
                end
            end
            %Check for finiteness
            for i = 1:numel(Phi.designMatrices)
                if(~all(all(all(isfinite(Phi.designMatrices{i})))))
                    warning('Non-finite design matrix Phi. Setting non-finite component to 0.')
                    Phi.designMatrices{i}(~isfinite(Phi.designMatrices{i})) = 0;
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(Phi.designMatrices{i}),...
                        find(~isfinite(Phi.designMatrices{i})))
                elseif(~all(all(all(isreal(Phi.designMatrices{i})))))
                    warning('Complex feature function output:')
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(Phi.designMatrices{i}),...
                        find(imag(Phi.designMatrices{i})))
                    disp('Ignoring imaginary part...')
                    Phi.designMatrices{i} = real(Phi.designMatrices{i});
                end
            end
            disp('done')
        end
        
        function Phi = standardizeDesignMatrix(Phi, featFuncMean, featFuncSqMean)
            %Standardize covariates to have 0 mean and unit variance
            disp('Standardize design matrix')
            %Compute std
            if(nargin > 1)
                Phi.featureFunctionStd = sqrt(featFuncSqMean - featFuncMean.^2);
            else
                Phi = Phi.computeFeatureFunctionMean;
                Phi = Phi.computeFeatureFunctionSqMean;
                Phi.featureFunctionStd = sqrt(Phi.featureFunctionSqMean - Phi.featureFunctionMean.^2);
                if(any(~isreal(Phi.featureFunctionStd)))
                    warning('Imaginary standard deviation. Setting it to 0.')
                    Phi.featureFunctionStd = real(Phi.featureFunctionStd);
                end
            end
            
            %centralize
            if(nargin > 1)
                for i = 1:numel(Phi.designMatrices)
                    Phi.designMatrices{i} = Phi.designMatrices{i} - featFuncMean;
                end
            else
                for i = 1:numel(Phi.designMatrices)
                    Phi.designMatrices{i} = Phi.designMatrices{i} - Phi.featureFunctionMean;
                end
            end
            
            %normalize
            for i = 1:numel(Phi.designMatrices)
                Phi.designMatrices{i} = Phi.designMatrices{i}./Phi.featureFunctionStd;
            end
            
            %Check for finiteness
            for i = 1:numel(Phi.designMatrices)
                if(~all(all(all(isfinite(Phi.designMatrices{i})))))
                    warning('Non-finite design matrix Phi. Setting non-finite component to 0.')
                    Phi.designMatrices{i}(~isfinite(Phi.designMatrices{i})) = 0;
                elseif(~all(all(all(isreal(Phi.designMatrices{i})))))
                    warning('Complex feature function output:')
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(Phi.designMatrices{i}),...
                        find(imag(Phi.designMatrices{i})))
                    disp('Ignoring imaginary part...')
                    Phi.designMatrices{i} = real(Phi.designMatrices{i});
                end
            end
            disp('done')
        end
        
        
        function Phi = normalizeDesignMatrix(Phi, normalizationFactors)
            %Normalize feature functions s.t. they lead to outputs of same magnitude.
            %This makes the likelihood gradient at theta_c = 0 better behaved.
            if(nargin > 1)
                for i = 1:numel(Phi.designMatrices)
                    Phi.designMatrices{i} = Phi.designMatrices{i}./normalizationFactors;
                end
            else
                for i = 1:numel(Phi.designMatrices)
                    Phi.designMatrices{i} = Phi.designMatrices{i}./Phi.featureFunctionAbsMean;
                end
            end
            for i = 1:numel(Phi.designMatrices)
                if(~all(all(all(isfinite(Phi.designMatrices{i})))))
                    warning('Non-finite design matrix Phi. Setting non-finite component to 0.')
                    Phi.designMatrices{i}(~isfinite(Phi.designMatrices{i})) = 0;
                elseif(~all(all(all(isreal(Phi.designMatrices{i})))))
                    warning('Complex feature function output:')
                    dataPoint = i
                    [coarseElement, featureFunction] = ind2sub(size(Phi.designMatrices{i}),...
                        find(imag(Phi.designMatrices{i})))
                    disp('Ignoring imaginary part...')
                    Phi.designMatrices{i} = real(Phi.designMatrices{i});
                end
            end
        end
        
        function Phi = saveNormalization(Phi, type)
            if(numel(Phi.featureFunctionMean) == 0)
                Phi = Phi.computeFeatureFunctionMean;
            end
            if(numel(Phi.featureFunctionSqMean) == 0)
                Phi = Phi.computeFeatureFunctionSqMean;
            end
            if strcmp(type, 'standardization')
                featureFunctionMean = Phi.featureFunctionMean;
                featureFunctionSqMean = Phi.featureFunctionSqMean;
                save('./data/featureFunctionMean', 'featureFunctionMean', '-ascii');
                save('./data/featureFunctionSqMean', 'featureFunctionSqMean', '-ascii');
            elseif strcmp(type, 'rescaling')
                featureFunctionMin = Phi.featureFunctionMin;
                featureFunctionMax = Phi.featureFunctionMax;
                save('./data/featureFunctionMin', 'featureFunctionMin', '-ascii');
                save('./data/featureFunctionMax', 'featureFunctionMax', '-ascii');
            else
                error('Which type of data normalization?')
            end
            
        end
        
        function Phi = computeSumPhiTPhi(Phi)
            Phi.sumPhiTPhi = 0;
            for i = 1:numel(Phi.dataSamples)
                Phi.sumPhiTPhi = Phi.sumPhiTPhi + Phi.designMatrices{i}'*Phi.designMatrices{i};
            end
        end
        
        
        
    end
    
end







