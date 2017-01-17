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
        
        function Phi = computeDesignMatrix(Phi, nElc, nElf)
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
            PhiCell{1} = zeros(nElc, nFeatureFunctions);
            PhiCell = repmat(PhiCell, nTrain, 1);
            parfor s = 1:nTrain
                %inputs belonging to same coarse element are in the same column of xk. They are ordered in
                %x-direction.
                PhiCell{s} = zeros(nElc, nFeatureFunctions);
                lambdak = zeros(nElf/nElc, nElc);
                for i = 1:nElc
                    lambdak(:, i) = conductivity{s}(coarseElement == i);
                end
                
                %construct design matrix Phi
                for i = 1:nElc
                    for j = 1:nFeatureFunctions
                        PhiCell{s}(i, j) = phi{j}(lambdak(:, i));
                    end
                end
            end
            
            Phi.designMatrices = PhiCell;
            for i = 1:nTrain
                if(~all(all(all(isfinite(Phi.designMatrices{i})))))
                    warning('Non-finite design matrix Phi. Setting non-finite component to 0.')
                    Phi.designMatrices{i}(~isfinite(Phi.designMatrices{i})) = 0;
                end
            end
            disp('done')
            Phi_computation_time = toc
            
        end
        
        function Phi = computeFeatureFunctionMean(Phi)
            %We normalize every feature function phi s.t. the mean output is 1 over the training set
            Phi.featureFunctionMean = 0;
            for i = 1:numel(Phi.designMatrices)
                Phi.featureFunctionMean = Phi.featureFunctionMean + mean(abs(Phi.designMatrices{i}), 1);
            end
            Phi.featureFunctionMean = Phi.featureFunctionMean/numel(Phi.designMatrices);
        end
        
        function Phi = computeFeatureFunctionSqMean(Phi)
            %We normalize every feature function phi s.t. the squares phi(x_k) for every macro-cell
            %k sum to 1
            featureFunctionSqSum = 0;
            for i = 1:numel(Phi.designMatrices)
                featureFunctionSqSum = featureFunctionSqSum + sum(Phi.designMatrices{i}.^2, 1);
            end
            Phi.featureFunctionSqMean = featureFunctionSqSum/...
                (numel(Phi.designMatrices)*size(Phi.designMatrices{1}, 1));
        end
        
        function Phi = standardizeDesignMatrix(Phi, featFuncMean, featFuncSqMean)
            %Standardize covariates to have 0 mean and unit variance
            
            %Compute std
            if(nargin > 1)
                Phi.featureFunctionStd = sqrt(featFuncSqMean - featFuncMean.^2);
            else
                Phi.featureFunctionStd = sqrt(Phi.featureFunctionSqMean - Phi.featureFunctionMean.^2);
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
                end
            end
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
                end
            end
        end
        
        function Phi = saveNormalization(Phi)
            if(numel(Phi.featureFunctionMean) == 0)
                Phi = Phi.computeFeatureFunctionMean;
            end
            if(numel(Phi.featureFunctionSqMean) == 0)
                Phi = Phi.computeFeatureFunctionSqMean;
            end
            featureFunctionMean = Phi.featureFunctionMean;
            featureFunctionSqMean = Phi.featureFunctionSqMean;
            save('./data/featureFunctionMean', 'featureFunctionMean', '-ascii');
            save('./data/featureFunctionSqMean', 'featureFunctionSqMean', '-ascii');
        end
        
        function Phi = computeSumPhiTPhi(Phi)
            Phi.sumPhiTPhi = 0;
            for i = 1:numel(Phi.dataSamples)
                Phi.sumPhiTPhi = Phi.sumPhiTPhi + Phi.designMatrices{i}'*Phi.designMatrices{i};
            end
        end
        
        
        
    end
    
end







