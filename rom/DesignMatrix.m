classdef DesignMatrix
    %Class describing the design matrices Phi for different data points
    
    properties
        
        designMatrices          %Design matrices stored in cells
        
        dataFile                %mat file holding the training/test data
        dataSamples             %vector with data sample indices
        
        featureFunctions        %Cell array of handles to feature functions
        featureFunctionAbsMean  %mean absolute output of feature function over training set BEFORE normalization
        
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
            parPoolInit(length(nTrain));
            PhiCell{1} = zeros(nElc, nFeatureFunctions);
            PhiCell = repmat(PhiCell, nTrain, 1);
            for s = 1:nTrain
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
                assert(all(all(isfinite(Phi.designMatrices{i}))), 'Error: non-finite design matrix Phi')
            end
            disp('done')
            Phi_computation_time = toc
            
        end
        
        function Phi = computeFeatureFunctionAbsMean(Phi)
            %We normalize every feature function phi s.t. the mean output is 1 over the training set
            Phi.featureFunctionAbsMean = 0;
            for i = 1:numel(Phi.designMatrices)
                Phi.featureFunctionAbsMean = Phi.featureFunctionAbsMean + mean(abs(Phi.designMatrices{i}), 1);
            end
            Phi.featureFunctionAbsMean = Phi.featureFunctionAbsMean/numel(Phi.designMatrices);
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
                assert(all(all(all(isfinite(Phi.designMatrices{i})))), 'Error: non-finite design matrix Phi')
            end
        end
        
        function Phi = saveNormalization(Phi)
            if(numel(Phi.featureFunctionAbsMean) == 0)
                Phi = Phi.computeFeatureFunctionAbsMean;
            end
            featureFunctionAbsMean = Phi.featureFunctionAbsMean;
            save('./data/featureFunctionAbsMean', 'featureFunctionAbsMean', '-ascii');
        end
        
        function Phi = computeSumPhiTPhi(Phi)
            Phi.sumPhiTPhi = 0;
            for i = 1:numel(Phi.dataSamples)
                Phi.sumPhiTPhi = Phi.sumPhiTPhi + Phi.designMatrices{i}'*Phi.designMatrices{i};
            end
        end
        
        
        
    end
    
end







