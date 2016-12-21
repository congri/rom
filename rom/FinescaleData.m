classdef FinescaleData
    %This class holds all the properties of the finescale data set and the methods to generate
    %the finescale data
    
    properties
        distributionType = 'correlated_binary';     %Type of p(x); usually use 'correlated_binary', which samples
                                                    %the conductivity field from a spacially
                                                    %correlated Gaussian
                                                    %Other possibilities:
                                                    %    'gaussian': Each pixel is drawn
                                                    %    independently from a Gaussian distribution
                                                    %    'uniform':  Each pixel is drawn
                                                    %    independently from a uniform distribution
                                                    %    'binary':   Each pixel is drawn
                                                    %    independently from a Bernoulli distribution
                                                    
        distributionParams                          %Cell array holding the distribution parameters:
                                                    %   'gaussian': first is mean, second is std
                                                    %   'uniform':  empty - upper and lower
                                                    %   conductivity are stored separately
                                                    %   'binary': volume fraction of high conducting
                                                    %   phase
                                                    %   'correlated_binary': volume fraction of high
                                                    %   conducting phase, length scale (vector with
                                                    %   x and y component, measured in pixels), sigma_f2
        
        nSets                                       %Number of data sets to generate;
                                                    %only difference is that different sets are
                                                    %stored to different .mat files
        
        nSamples                                    %Vector giving the number of samples per set
        
        loCond                                      %Lower conductivity
        upCond                                      %Upper conductivity
        contrast                                    %contrast, i.e. upper/lower
    end
    
    methods
        
        %constructor
        function FD = FinescaleData(loCond, upCond)
            FD.loCond = loCond;
            FD.upCond = upCond;
            FD.contrast = upCond/loCond;
        end
        
        function cond = generateConductivityField(FD, domain, nSet)
            %nSet is the number of the data set
            
            % Draw conductivity/ log conductivity
            disp('Generating finescale conductivity field...')
            tic
            if strcmp(FD.distributionType, 'uniform')
                %conductivity uniformly distributed between lo and up
                cond{1} = zeros(domain.nEl, 1);
                cond = repmat(cond, 1, FD.nSamples(nSet));
                for i = 1:FD.nSamples(nSet)
                    cond{i} = (FD.upCond - FD.loCond)*rand(domain.nEl, 1) + FD.loCond;
                end
            elseif strcmp(FD.distributionType, 'gaussian')
                %log conductivity gaussian distributed
                x = normrnd(FD.distributionParams{1}, FD.distributionParams{2}, domain.nEl, FD.nSamples(nSet));
                cond{1} = zeros(domain.nEl, 1);
                cond = repmat(cond, 1, FD.nSamples(nSet));
                for i = 1:FD.nSamples(nSet)
                    cond{i} = exp(x(:, i));
                end
            elseif strcmp(FD.distributionType, 'binary')
                %binary distribution of conductivity (Bernoulli)
                cond{1} = zeros(domain.nEl, 1);
                cond = repmat(cond, 1, FD.nSamples(nSet));
                for i = 1:FD.nSamples(nSet)
                    r = rand(domain.nEl, 1);
                    cond{i} = FD.loCond*ones(domain.nEl, 1);
                    cond{i}(r < FD.distributionParams{1}) = FD.upCond;
                end
            elseif strcmp(FD.distributionType, 'correlated_binary')
                %ATTENTION: so far, only isotropic distributions (length scales) possible
                %Compute coordinates of element centers
                x = (domain.lElX/2):domain.lElX:(1 - (domain.lElX/2));
                y = (domain.lElY/2):domain.lElY:(1 - (domain.lElY/2));
                [X, Y] = meshgrid(x, y);
                %directly clear potentially large arrays
                clear y;
                x = [X(:) Y(:)]';
                clear X Y;
                
                parPoolInit(FD.nSamples(nSet));
                %Store conductivity fields in cell array to avoid broadcasting the whole data
                cond{1} = zeros(domain.nEl, 1);
                cond = repmat(cond, 1, FD.nSamples(nSet));
                
                nBochnerBasis = 1e3;    %Number of cosine basis functions
                for i = 1:(FD.nSamples(nSet))
                    p{i} = genBochnerSamples(domain.lElX*FD.distributionParams{2}(1),...
                        FD.distributionParams{3}, nBochnerBasis);
                end
                nEl = domain.nEl;
                cutoff = norminv(1 - FD.distributionParams{1}, 0, FD.distributionParams{3});
                parfor i = 1:(FD.nSamples(nSet))
                    %use for-loop instead of vectorization to save memory
                    for j = 1:nEl
                        ps = p{i}(x(:, j));
                        cond{i}(j) = FD.upCond*(ps > cutoff) +...
                            FD.loCond*(ps <= cutoff);
                    end
                end
            else
                error('unknown FOM conductivity distribution');
            end
            disp('done')
            toc
        end
        
        function Tf = solveFEM(FD, domain, nSet, savepath)
            
            cond = FD.generateConductivityField(domain, nSet);
            %Solve finite element model
            disp('Solving finescale problem...')
            tic
            Tf = zeros(domain.nNodes, FD.nSamples(nSet));
            D{1} = zeros(2, 2, domain.nEl);
            D = repmat(D, FD.nSamples(nSet), 1);
            nEl = domain.nEl;   %To avoid broadcasting overhead
            parPoolInit(FD.nSamples(nSet));
            parfor i = 1:FD.nSamples(nSet)
                %Conductivity matrix D, only consider isotropic materials here
                for j = 1:nEl
                    D{i}(:, :, j) =  cond{i}(j)*eye(2);
                end
                FEMout = heat2d(domain, D{i});
                %Store fine temperatures as a vector Tf. Use reshape(Tf(:, i), domain.nElX + 1, domain.nElY + 1)
                %and then transpose result to reconvert it to original temperature field
                Ttemp = FEMout.Tff';
                Tf(:, i) = Ttemp(:);
            end
            disp('FEM systems solved')
            toc
            
            if(nargin > 2)
                disp('saving finescale data...')
                cond = cell2mat(cond);
                save(strcat(savepath, ''), 'cond', 'Tf', '-v7.3')    %partial loading only for -v7.3
                disp('done')
            end
        end
    end
    
end







