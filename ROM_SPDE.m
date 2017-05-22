classdef ROM_SPDE
    %Class for ROM SPDE model
    
    properties
        %% Finescale data specifications
        %Finescale system size
        nElFX = 256;
        nElFY = 256;
        %Finescale conductivities, binary material
        lowerConductivity = 1;
        upperConductivity = 10;
        %Finescale total number of samples in set 1, needed to load data
        totalFinescaleSamples = 1024;
        %Correlation length in number of pixels
        correlationLength = 20;
        %High conducting phase volume fraction
        volumeFraction = .2;
        %Sigma parameter of squared exponential kernel of Gaussian random field for finescale
        %microstructure
        sigma_f2 = 1;
        %Conductivity field distribution type
        conductivityDistribution = 'correlated_binary';
        %Coefficients giving boundary conditions, specify as string
        boundaryConditions = '[0 1000 0 0]';
        %Boundary condition functions; evaluate those on boundaries to get boundary conditions
        boundaryTemperature;
        boundaryHeatFlux;
        %Directory where finescale data is stored; specify basename here
        finescaleDataPath = '~/matlab/data/fineData/';
        %matfile handle
        finescaleDataMatfile;
        %Finescale Domain object
        fineScaleDomain;
        
        %% Coarse model specifications
        coarseScaleDomain;
        coarseGridVectorX = [.5 .5];
        coarseGridVectorY = [.5 .5];
        
    end
    
    methods
        function obj = loadTrainingData(obj)
            obj.finescaleDataPath = strcat(obj.finescaleDataPath,...
                'systemSize=', num2str(obj.nElFX), 'x', num2str(obj.nElFY), '/');
            %Type of conductivity distribution
            if strcmp(obj.conductivityDistribution, 'correlated_binary')
                obj.finescaleDataPath = strcat(obj.finescaleDataPath,...
                    obj.conductivityDistribution, '/', 'IsoSEcov/', 'l=',...
                    num2str(obj.correlationLength), '_sigmafSq=', num2str(obj.sigma_f2),...
                    '/volumeFraction=', num2str(obj.volumeFraction), '/', 'locond=',...
                    num2str(obj.lowerConductivity), '_upcond=', num2str(obj.upperConductivity),...
                    '/', 'BCcoeffs=', obj.boundaryConditions, '/');
            elseif strcmp(cond_distribution, 'binary')
                obj.finescaleDataPath = strcat(obj.finescaleDataPath,...
                    obj.conductivityDistribution, '/volumeFraction=',...
                    num2str(obj.volumeFraction), '/', 'locond=', num2str(obj.lowerConductivity),...
                    '_upcond=', num2str(obj.upperConductivity), '/', 'BCcoeffs=', obj.boundaryConditions, '/');
            else
                error('Unknown conductivity distribution')
            end            
            
            %Name of training data file
            trainFileName = strcat('set1-samples=', num2str(obj.totalFinescaleSamples), '.mat');
            %Name of parameter file
            paramFileName = strcat('params','.mat');
            
            %Set up boundary condition functions
            bc = str2num(obj.boundaryConditions);
            obj.boundaryTemperature = @(x) bc(1) + bc(2)*x(1) + bc(3)*x(2) + bc(4)*x(1)*x(2);
            obj.boundaryHeatFlux{1} = @(x) -(bc(3) + bc(4)*x);      %lower bound
            obj.boundaryHeatFlux{2} = @(y) (bc(2) + bc(4)*y);       %right bound
            obj.boundaryHeatFlux{3} = @(x) (bc(3) + bc(4)*x);       %upper bound
            obj.boundaryHeatFlux{4} = @(y) -(bc(2) + bc(4)*y);      %left bound

            %for finescale domain class
            addpath('./heatFEM')
            %for boundary condition functions
            %load data params; warning for variable FD can be ignored
            load(strcat(obj.finescaleDataPath, paramFileName));
            
            %domainf is outdated notation
            obj.fineScaleDomain = domainf;
            clear domainf;
            
            %there is no cum_lEl (cumulated finite element length) in old data files
            if(~numel(obj.fineScaleDomain.cum_lElX) || ~numel(obj.fineScaleDomain.cum_lElX))
                obj.fineScaleDomain.cum_lElX = linspace(0, 1, obj.fineScaleDomain.nElX + 1);
                obj.fineScaleDomain.cum_lElY = linspace(0, 1, obj.fineScaleDomain.nElY + 1);
            end
            
            %load finescale temperatures partially
            obj.finescaleDataMatfile = matfile(strcat(obj.finescaleDataPath, trainFileName));  
        end
        
        
        
        
        function obj = genCoarseDomain(obj)
            %Generate coarse domain object
            nX = length(obj.coarseGridVectorX);
            nY = length(obj.coarseGridVectorY);
            obj.coarseScaleDomain = Domain(nX, nY, obj.coarseGridVectorX, obj.coarseGridVectorY);
            %ATTENTION: natural nodes have to be set manually
            %and consistently in coarse and fine scale domain!!
            obj.coarseScaleDomain = setBoundaries(obj.coarseScaleDomain, [2:(2*nX + 2*nY)],...
                obj.boundaryTemperature, obj.boundaryHeatFlux);
            
            %Legacy, for predictions
            if ~exist('./data/', 'dir')
                mkdir('./data/');
            end
            filename = './data/domainc.mat';
            save(filename, 'obj.coarseScaleDomain');
        end
    end
    
end


















