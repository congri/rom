classdef DeterministicBinaryAutoencoder
    
    properties
        trainingData        %must be binary
        latentDim = 6;
        
        %Model parameters
        Wz              %Projection from x to z, i.e. dim(z) x dim(x)
        bz              %Bias from x to z, dim(z)
        Wx              %Projection from z back to x, dim(x) x dim(z)
        bx              %Bias from z to x, dim(x)
        
        mode = 'linear' %'sigmoid' for sigmoid transform, 'linear' for only linear mapping
        
        %Convergence criteria
        maxIterations = 100;
    end
    
    methods
        function [r, d_r] = residual(this, paramsVec, dim, latentDim, bx)
            
            Wx = reshape(paramsVec(1:(dim*latentDim)), dim, latentDim);
            Wz = reshape(paramsVec((dim*latentDim + 1):(2*dim*latentDim)), latentDim, dim);
            bz = paramsVec((2*dim*latentDim + 1):end);

            %Latent variable projection
            z = Wz*this.trainingData + bz;
            %n-th column belongs to n-th data point
            A = Wx*z + bx;
            if strcmp(this.mode, 'sigmoid')
                x_model = sigmoid(A);
            else
                x_model = A;
            end
            diff = x_model - this.trainingData;
            r = sum(sum(diff.^2));
            
            if nargout > 1
                if strcmp(this.mode, 'sigmoid')
                    dSigma_dA = x_model.*(1 - x_model);
                    two_diff_times_dSigma_dA = 2*diff.*dSigma_dA;
                    dF_dbx = sum(two_diff_times_dSigma_dA, 2);
                    dF_dbz = Wx'*dF_dbx;
                    dF_dWx = two_diff_times_dSigma_dA*z';
                    dF_dWz = Wx'*two_diff_times_dSigma_dA*this.trainingData';
                else
                    two_diff = 2*diff;
                    dF_dbx = sum(two_diff, 2);
                    dF_dbz = Wx'*dF_dbx;
                    dF_dWx = two_diff*z';
                    dF_dWz = Wx'*two_diff*this.trainingData';
                end
                
                %Projections first
                d_r = [dF_dWx(:); dF_dWz(:); dF_dbz];
            end
            
        end
        
        
        
        function this = train(this, paramsVec)
            %params_0 are parameter start values
            
            ppool = parPoolInit(16);
            
            
            dim = size(this.trainingData, 1);
            latentDim = this.latentDim;
            if nargin < 2
                rng(2) %For repoducability
                paramsVec = 4*rand(2*dim*latentDim + latentDim, 1) - 2;
            end
            this.Wx = reshape(paramsVec(1:(dim*latentDim)), dim, latentDim);
            this.Wz = reshape(paramsVec((dim*latentDim + 1):(2*dim*latentDim)), latentDim, dim);
            this.bz = paramsVec((2*dim*latentDim + 1):end);
            
            trainingDataMean = mean(this.trainingData, 2);
            
            for i = 1:10
                this.bx = trainingDataMean - this.Wx*this.bz - this.Wx*this.Wz*trainingDataMean;
                
                fun = @(params) this.residual(params, dim, this.latentDim, this.bx);
                options = optimoptions('fminunc', 'Algorithm', 'quasi-newton',...
                    'SpecifyObjectiveGradient', true, 'MaxIterations', this.maxIterations, ...
                    'FunctionTolerance', 1e-10, 'StepTolerance', 1e-10,...
                    'Display', 'Iter-detailed', 'HessUpdate', 'dfp', 'UseParallel', true);
                [paramsVec, res] = fminunc(fun, paramsVec, options);
                
                
                %Roughly the fraction of miss-predicted pixels
                selfErr = sqrt(res)/numel(this.trainingData)
                this.Wx = reshape(paramsVec(1:(dim*latentDim)), dim, latentDim);
                this.Wz = reshape(paramsVec((dim*latentDim + 1):(2*dim*latentDim)), latentDim, dim);
                this.bz = paramsVec((2*dim*latentDim + 1):end);
            end
        end
        
        
        
        function [decodedData] = decode(this, encodedData)
            decodedData = this.Wx*encodedData + this.bx;
        end
        
        
        
        function [encodedData] = encode(this, originalData)
            encodedData = this.Wz*originalData + this.bz;
        end
        
        
        
        function err = reconstructionErr(this, decodedData, trueData)
            %Gives fraction of falsely reconstructed pixels
            err = sum(sum(abs(trueData - decodedData)))/numel(trueData);
        end
    end
    
end

