classdef DesignMatrix
    %Class describing the design matrices Phi for different data points
    
    properties
        
        designMatrices          %Design matrices stored in cells
        
        dataFile                %mat file holding the training/test data
        dataSamples             %vector with data sample indices
        
        featureFunctions        %Cell array of handles to local feature functions
        globalFeatureFunctions = [];  %feature functions that take the whole microstructure as input
        featureFunctionMean  %mean absolute output of feature function over training set BEFORE normalization
        featureFunctionSqMean
        featureFunctionStd
        featureFunctionMin
        featureFunctionMax
        
        E                       %gives the coarse element a fine element belongs to
        EMat                    %fine to coarse index map as a matrix
        lambdak                 %conductivities in macro-cell k
        xk                      %transformed conductivities in macro-cell k
        transformedConductivity %transformed conductivity matrices
        sumPhiTPhi
        
        neighborDictionary      %This array holds the index of theta,
                                %the corresponding feature function number, coarse element and neighboring number
        useAutoEnc
        latentDim = 0;          %If autoencoder is used
        
    end
    
    methods
        
        %constructor
        function Phi = DesignMatrix(domainf, domainc, featureFunctions,...
                globalFeatureFunctions, dataFile, dataSamples)
            %Set up mapping from fine to coarse element
            Phi = Phi.getCoarseElement(domainc, domainf);
            Phi.featureFunctions = featureFunctions;
            Phi.globalFeatureFunctions = globalFeatureFunctions;
            Phi.dataFile = dataFile;
            Phi.dataSamples = dataSamples;
            
        end
        
        function Phi = getCoarseElement(Phi, domainc, domainf)
            Phi.E = zeros(domainf.nEl, 1);
            e = 1;  %element number
            for row_fine = 1:domainf.nElY
                %coordinate of lower boundary of fine element
                y_coord = domainf.cum_lElY(row_fine);
                row_coarse = sum(y_coord >= domainc.cum_lElY);
                for col_fine = 1:domainf.nElX
                    %coordinate of left boundary of fine element
                    x_coord = domainf.cum_lElX(col_fine);
                    col_coarse = sum(x_coord >= domainc.cum_lElX);
                    Phi.E(e) = (row_coarse - 1)*domainc.nElX + col_coarse;
                    e = e + 1;
                end
            end
            
            Phi.EMat = reshape(Phi.E, domainf.nElX, domainf.nElY);
            pltFineToCoarse = false;
            if pltFineToCoarse
                figure
                imagesc(Phi.EMat)
                pause
            end
        end
        
        function [lambdak, xk] = get_coarseElementConductivities(Phi, nElc, nElf, condTransOpts)
            %load finescale conductivity field
            conductivity = Phi.dataFile.cond(:, Phi.dataSamples);
            nTrain = length(Phi.dataSamples);
            
            %Open parallel pool
%             addpath('./computation')
%             parPoolInit(nTrain);
            EMatHold = Phi.EMat;
            lambdak = cell(nTrain, nElc);
            xk = cell(nTrain, nElc);
            
            for s = 1:nTrain
                %inputs belonging to same coarse element are in the same column of xk. They are ordered in
                %x-direction.
                %Get conductivity fields in coarse cell windows
                conductivityMat = reshape(conductivity(:, s), sqrt(nElf), sqrt(nElf));
                for i = 1:nElc
                    indexMat = (EMatHold == i);
                    lambdakTemp = conductivityMat.*indexMat;
                    %Cut elements from matrix that do not belong to coarse cell
                    lambdakTemp(~any(lambdakTemp, 2), :) = [];
                    lambdakTemp(:, ~any(lambdakTemp, 1)) = [];
                    lambdak{s, i} = lambdakTemp;
                    xk{s, i} = conductivityTransform(lambdak{s, i}, condTransOpts);
                end
            end
        end
        
        function Phi = computeDesignMatrix(Phi, nElc, nElf, condTransOpts)
            %Actual computation of design matrix
            debug = false; %for debug mode
            tic
            disp('Compute design matrices Phi...')
            
            %load finescale conductivity field
            conductivity = Phi.dataFile.cond(:, Phi.dataSamples);
            conductivity = num2cell(conductivity, 1);   %to avoid parallelization communication overhead
            nTrain = length(Phi.dataSamples);
            nFeatureFunctions = size(Phi.featureFunctions, 2);
            Phi.globalFeatureFunctions;
            nGlobalFeatureFunctions = size(Phi.globalFeatureFunctions, 2);
            phi = Phi.featureFunctions;
            phiGlobal = Phi.globalFeatureFunctions;
%             coarseElement = Phi.E;
            
            %Open parallel pool
%             addpath('./computation')
%             parPoolInit(nTrain);
            PhiCell{1} = zeros(nElc, nFeatureFunctions + nGlobalFeatureFunctions);
            PhiCell = repmat(PhiCell, nTrain, 1);
            EMatHold = Phi.EMat;
            [lambdak, xk] = Phi.get_coarseElementConductivities(nElc, nElf, condTransOpts);
            Phi.lambdak = lambdak;
            Phi.xk = xk;
            
%             transformedConductivityHold = cell(nTrain, 1);
            
            if Phi.useAutoEnc
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
                %This is wrong for purely high conducting microstructures!!!
                loCond = min(min(lambdakMat));
                lambdakMatBin = logical(lambdakMat - loCond);
                %Encoded version of test samples
                load('./autoencoder/trainedAutoencoder.mat');
                latentMu = ba.encode(lambdakMatBin);
                Phi.latentDim = ba.latentDim;
                if ~debug
                    clear ba;
                end
                latentMu = reshape(latentMu, Phi.latentDim, nElc, nTrain);
            end


%             parfor s = 1:nTrain
            for s = 1:nTrain    %for very cheap features, serial evaluation might be more efficient
                %inputs belonging to same coarse element are in the same column of xk. They are ordered in
                %x-direction.
                %Get conductivity fields in coarse cell windows
                conductivityMat = reshape(conductivity{s}, sqrt(nElf), sqrt(nElf));
%                 transformedConductivityHold{s} = conductivityTransform(conductivityMat, condTransOpts);

                %construct design matrix Phi
                for i = 1:nElc
                    %local features
                    for j = 1:nFeatureFunctions
                        %only take pixels of corresponding macro-cell as input for features
                        PhiCell{s}(i, j) = phi{i, j}(lambdak{s, i});
                    end
                    %global features
                    for j = 1:nGlobalFeatureFunctions
                        %Take whole microstructure as input for feature function
                        PhiCell{s}(i, nFeatureFunctions + j) = phiGlobal{i, j}(conductivityMat);
                    end
                    if Phi.useAutoEnc
                        for j = 1:Phi.latentDim
                            PhiCell{s}(i, nFeatureFunctions + nGlobalFeatureFunctions + j) = latentMu(j, i, s);
                        end
                    end
                end
            end
            
            if debug
                for n = 1:nTrain
                    for k = 1:nElc
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
            
%             Phi.transformedConductivity = transformedConductivityHold;
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
        
        
        
        
        function Phi = secondOrderFeatures(Phi, A)
            %Includes second order multinomial terms, i.e. a_ij phi_i phi_j, where a_ij is logical.
            %Squared term phi_i^2 if a_ii ~= 0. To be executed directly after feature function
            %computation.
            %Input A is the logical array giving the squared terms to include
            
            assert(all(all(islogical(A))), 'A must be a logical array of nFeatures x nFeatures')
            %Consider every term only once
            assert(sum(sum(tril(A, -1))) == 0, 'Matrix A must be upper triangular')
            
            disp('Using squared terms of feature functions...')
            nElc = size(Phi.designMatrices{1}, 1);
            nFeatureFunctions = size(Phi.featureFunctions, 2);
            nSecondOrderTerms = sum(sum(A));
            PhiCell{1} = zeros(nElc, nSecondOrderTerms + nFeatureFunctions);
            nTrain = length(Phi.dataSamples);
            PhiCell = repmat(PhiCell, nTrain, 1);
            
            for s = 1:nTrain
                %The first columns contain first order terms
                PhiCell{s}(:, 1:nFeatureFunctions) = Phi.designMatrices{s};
                
                %Second order terms
                f = 1;
                for r = 1:size(A, 1)
                    for c = r:size(A, 2)
                        if A(r, c)
                            PhiCell{s}(:, nFeatureFunctions + f) = ...
                                PhiCell{s}(:, r).*PhiCell{s}(:, c);
                            f = f + 1;
                        end
                    end
                end
            end
            Phi.designMatrices = PhiCell;
            disp('done')
        end%secondOrderFeatures
        
        
        
        
        
        
        function Phi = includeNearestNeighborFeatures(Phi, nc)
            %Includes feature function information of neighboring cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest neighbor feature function information...')
            nElc = prod(nc);
            nFeatureFunctionsTotal = size(Phi.designMatrices{1}, 2);
            PhiCell{1} = zeros(nElc, 5*nFeatureFunctionsTotal);
            nTrain = length(Phi.dataSamples);
            PhiCell = repmat(PhiCell, nTrain, 1);
            
            for s = 1:nTrain
                %The first columns contain feature function information of the original cell
                PhiCell{s}(:, 1:nFeatureFunctionsTotal) = Phi.designMatrices{s};
                
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                for i = 1:nElc
                    if(mod(i, nc(1)) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{s}(i, (nFeatureFunctionsTotal + 1):(2*nFeatureFunctionsTotal)) =...
                           Phi.designMatrices{s}(i + 1, :);
                    end
                    
                    if(i <= nc(1)*(nc(2) - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{s}(i, (2*nFeatureFunctionsTotal + 1):(3*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i + nc(1), :);
                    end
                    
                    if(mod(i - 1, nc(1)) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{s}(i, (3*nFeatureFunctionsTotal + 1):(4*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i - 1, :);
                    end
                    
                    if(i > nc(1))
                        %lower neighbor of coarse element exists
                        PhiCell{s}(i, (4*nFeatureFunctionsTotal + 1):(5*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i - nc(1), :);
                    end
                end
            end
            Phi.designMatrices = PhiCell;
            disp('done')
        end%includeNearestNeighborFeatures
        
        
        function Phi = includeLocalNearestNeighborFeatures(Phi, nc)
            %Includes feature function information of neighboring cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest neighbor feature function information separately for each cell...')
            nElc = prod(nc);
            nFeatureFunctionsTotal = size(Phi.designMatrices{1}, 2);
            nTrain = numel(Phi.dataSamples);
            
            for s = 1:nTrain
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                k = 0;
                for i = 1:nElc
                    PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                        Phi.designMatrices{s}(i, :);
                    Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                        (1:nFeatureFunctionsTotal)'; %feature index
                    Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                        i; %coarse element index
                    Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                        0; %center element
                    k = k + 1;
                    if(mod(i, nc(1)) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i + 1, :);
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            1; %right neighbor
                        k = k + 1;
                    end
                    
                    if(i <= nc(1)*(nc(2) - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i + nc(1), :);
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            2; %upper neighbor
                        k = k + 1;
                    end
                    
                    if(mod(i - 1, nc(1)) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i - 1, :);
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            3; %left neighbor
                        k = k + 1;
                    end
                    
                    if(i > nc(1))
                        %lower neighbor of coarse element exists
                        PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i - nc(1), :);
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            4; %lower neighbor
                        k = k + 1;
                    end
                end
            end
            Phi.designMatrices = PhiCell;
            disp('done')
        end%includeLocalNearestNeighborFeatures
        
        
        function Phi = includeDiagNeighborFeatures(Phi, nc)
            %includes feature function information of all other cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest and diagonal neighbor feature function information...')
            nElc = prod(nc);
            nFeatureFunctions = size(Phi.designMatrices{1}, 2);
            PhiCell{1} = zeros(nElc, 9*nFeatureFunctions);
            nTrain = length(Phi.dataSamples);
            PhiCell = repmat(PhiCell, nTrain, 1);
            
            for s = 1:nTrain
                %The first columns contain feature function information of the original cell
                PhiCell{s}(:, 1:nFeatureFunctions) = Phi.designMatrices{s};
                
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                for i = 1:nElc
                    if(mod(i, nc(1)) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{s}(i, (nFeatureFunctions + 1):(2*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i + 1, :);
                        if(i <= nc(1)*(nc(2) - 1))
                            %upper right neighbor of coarse element exists
                            PhiCell{s}(i, (2*nFeatureFunctions + 1):(3*nFeatureFunctions)) =...
                                Phi.designMatrices{s}(i + nc(1) + 1, :);
                        end
                    end
                    
                    if(i <= nc(1)*(nc(2) - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{s}(i, (3*nFeatureFunctions + 1):(4*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i + nc(1), :);
                        if(mod(i - 1, nc(1)) ~= 0)
                            %upper left neighbor exists
                            PhiCell{s}(i, (4*nFeatureFunctions + 1):(5*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i + nc(1) - 1, :);
                        end
                    end
                    
                    if(mod(i - 1, nc(1)) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{s}(i, (5*nFeatureFunctions + 1):(6*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i - 1, :);
                        if(i > nc(1))
                            %lower left neighbor exists
                            PhiCell{s}(i, (6*nFeatureFunctions + 1):(7*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i - nc(1) - 1, :);
                        end
                    end
                    
                    if(i > nc(1))
                        %lower neighbor of coarse element exists
                        PhiCell{s}(i, (7*nFeatureFunctions + 1):(8*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i - nc(1), :);
                        if(mod(i, nc(1)) ~= 0)
                            %lower right neighbor exists
                            PhiCell{s}(i, (8*nFeatureFunctions + 1):(9*nFeatureFunctions)) =...
                            Phi.designMatrices{s}(i - nc(1) + 1, :);
                        end
                    end
                end
            end
            Phi.designMatrices = PhiCell;
            disp('done')
        end%includeDiagNeighborFeatures
        
        
        
        function Phi = includeLocalDiagNeighborFeatures(Phi, nc)
            %Includes feature function information of direct and diagonal neighboring cells
            %Can only be executed after standardization/rescaling!
            %nc/nf: coarse/fine elements in x/y direction
            disp('Including nearest + diagonal neighbor feature function information separately for each cell...')
            nElc = prod(nc);
            nFeatureFunctionsTotal = size(Phi.designMatrices{1}, 2);
%             PhiCell{1} = zeros(nElc, 5*nFeatureFunctions);
            nTrain = length(Phi.dataSamples);
%             PhiCell = repmat(PhiCell, nTrain, 1);
            
            for s = 1:nTrain
                %Only assign nonzero values to design matrix for neighboring elements if
                %neighbor in respective direction exists
                k = 0;
                for i = 1:nElc
                    PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                        Phi.designMatrices{s}(i, :);
                    Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                        (1:nFeatureFunctionsTotal)'; %feature index
                    Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                        i; %coarse element index
                    Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                        0; %center element
                    k = k + 1;
                    if(mod(i, nc(1)) ~= 0)
                        %right neighbor of coarse element exists
                        PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i + 1, :);
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            1; %right neighbor
                        k = k + 1;
                        
                        if(i <= nc(1)*(nc(2) - 1))
                            %upper right neighbor of coarse element exists
                            PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                                Phi.designMatrices{s}(i + nc(1) + 1, :);
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                                (1:nFeatureFunctionsTotal)'; %feature index
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                                i; %coarse element index
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                                2; % upper right neighbor
                            k = k + 1;
                        end
                        
                    end
                    
                    
                    if(i <= nc(1)*(nc(2) - 1))
                        %upper neighbor of coarse element exists
                        PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i + nc(1), :);
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            2; %upper neighbor
                        k = k + 1;
                        
                        if(mod(i - 1, nc(1)) ~= 0)
                            %upper left neighbor of coarse element exists
                            PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                                Phi.designMatrices{s}(i + nc(1) - 1, :);
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                                (1:nFeatureFunctionsTotal)'; %feature index
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                                i; %coarse element index
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                                4; % upper left neighbor
                            k = k + 1;
                        end
                        
                    end
                    
                    
                    if(mod(i - 1, nc(1)) ~= 0)
                        %left neighbor of coarse element exists
                        PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i - 1, :);
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            3; %left neighbor
                        k = k + 1;
                        
                        if(i > nc(1))
                            %lower left neighbor of coarse element exists
                            PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                                Phi.designMatrices{s}(i - nc(1) - 1, :);
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                                (1:nFeatureFunctionsTotal)'; %feature index
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                                i; %coarse element index
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                                6; % lower left neighbor
                            k = k + 1;
                        end
                        
                    end
                    
                    
                    if(i > nc(1))
                        %lower neighbor of coarse element exists
                        PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                            Phi.designMatrices{s}(i - nc(1), :);
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                            (1:nFeatureFunctionsTotal)'; %feature index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                            i; %coarse element index
                        Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                            4; %lower neighbor
                        k = k + 1;
                        
                        if(mod(i, nc(1)) ~= 0)
                            %lower right neighbor of coarse element exists
                            PhiCell{s}(i, (k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal)) =...
                                Phi.designMatrices{s}(i - nc(1) + 1, :);
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 1) = ...
                                (1:nFeatureFunctionsTotal)'; %feature index
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 2) = ...
                                i; %coarse element index
                            Phi.neighborDictionary((k*nFeatureFunctionsTotal + 1):((k + 1)*nFeatureFunctionsTotal), 3) = ...
                                8; % lower right neighbor
                            k = k + 1;
                        end
                        
                    end
                end
            end
            Phi.designMatrices = PhiCell;
            disp('done')
        end%includeLocalDiagNeighborFeatures
        
        
        function Phi = localTheta_c(Phi, nc)
            %Sets separate coefficients theta_c for each macro-cell in a single microstructure
            %sample
            %Can never be executed before rescaling/standardization of design Matrix!
            debug = false; %debug mode
            disp('Using separate feature coefficients theta_c for each macro-cell in a microstructure...')
            nElc = prod(nc);
            nFeatureFunctionsTotal = size(Phi.designMatrices{1}, 2);
            PhiCell{1} = zeros(nElc, nElc*nFeatureFunctionsTotal);
            nTrain = length(Phi.dataSamples);
            PhiCell = repmat(PhiCell, nTrain, 1);
            
            %Reassemble design matrix
            for s = 1:nTrain
                for i = 1:nElc
                    PhiCell{s}(i, ((i - 1)*nFeatureFunctionsTotal + 1):(i*nFeatureFunctionsTotal)) = ...
                      Phi.designMatrices{s}(i, :);
                end
                PhiCell{s} = sparse(PhiCell{s});
            end
            if debug
                firstDesignMatrixBeforeLocal = Phi.designMatrices{1}
                firstDesignMatrixAfterLocal = full(PhiCell{1})
                pause
            end
            Phi.designMatrices = PhiCell;
            disp('done')
        end%localTheta_c
        
        function Phi = computeFeatureFunctionMinMax(Phi)
            %Computes min/max of feature function outputs over training data, separately for every
            %macro cell
            Phi.featureFunctionMin = Phi.designMatrices{1};
            Phi.featureFunctionMax = Phi.designMatrices{1};
            for i = 1:numel(Phi.designMatrices)
                Phi.featureFunctionMin(Phi.featureFunctionMin > Phi.designMatrices{i}) =...
                    Phi.designMatrices{i}(Phi.featureFunctionMin > Phi.designMatrices{i});
                Phi.featureFunctionMax(Phi.featureFunctionMax < Phi.designMatrices{i}) =...
                    Phi.designMatrices{i}(Phi.featureFunctionMax < Phi.designMatrices{i});
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
                featFuncDiff = featFuncMax - featFuncMin;
                %to avoid irregularities due to rescaling (if every macro cell has the same feature function output)
                featFuncMin(featFuncDiff == 0) = 0;
                featFuncDiff(featFuncDiff == 0) = 1;
                for i = 1:numel(Phi.designMatrices)
                    Phi.designMatrices{i} = (Phi.designMatrices{i} - featFuncMin)./...
                        (featFuncDiff);
                end
            else
                Phi = Phi.computeFeatureFunctionMinMax;
                featFuncDiff = Phi.featureFunctionMax - Phi.featureFunctionMin;
                %to avoid irregularities due to rescaling (if every macro cell has the same feature function output)
                Phi.featureFunctionMin(featFuncDiff == 0) = 0;
                featFuncDiff(featFuncDiff == 0) = 1;
                for i = 1:numel(Phi.designMatrices)
                    Phi.designMatrices{i} = (Phi.designMatrices{i} - Phi.featureFunctionMin)./...
                        (featFuncDiff);
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
        
        
        
        
        
        function saveNormalization(Phi, type)
            disp('Saving Phi-normalization...')
            if(isempty(Phi.featureFunctionMean))
                Phi = Phi.computeFeatureFunctionMean;
            end
            if(isempty(Phi.featureFunctionSqMean))
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
        
        
        
        
        function Phi = addLinearFilter(Phi, w)
            %w is a cell array with numel(w) = nEl_coarse holding the linear filter for every
            %macro-cell
            %Can be done more efficiently!
            
            %construct design matrix Phi
            nTrain = numel(Phi.dataSamples);
            nElc = numel(w);
            nFeaturesTotalAfter = size(Phi.designMatrices{1}, 1) + nElc;
            PhiCell{1} = zeros(size(Phi.designMatrices{1}, 1), nFeaturesTotalAfter);
            PhiCell = repmat(PhiCell, nTrain, 1);
            n = 1;
            for s = 1:nTrain
                for m = 1:nElc
                    for j = 1:nFeaturesTotalAfter
                        if(j == m*(nFeaturesTotalAfter/nElc))
%                             PhiCell{s}(m, j) = sum(w{m}'.*log(Phi.lambdak{s, m}(:)));
                            PhiCell{s}(m, j) = Phi.featureFunctions{m, end}(Phi.lambdak{s, m});
                        elseif(mod(j, nFeaturesTotalAfter/nElc))
                            %the elseif is due to skipping of indices of wrong coarse elements
                            PhiCell{s}(m, j) = Phi.designMatrices{s}(m, n);
                            n = n + 1;
                        end
                    end
                    n = 1;
                end
            end
            Phi.designMatrices = PhiCell;
        end
        
        
        
    end
    
end







