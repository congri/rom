classdef DeterministicBinaryAutoencoder
    
    properties
        trainingData        %must be binary
        latentDim = 10;
        
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
        function [r, d_r] = residual(this, bx, bz, Wx, Wz)
            
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
            
%             if nargout > 1
%                 if strcmp(this.mode, 'sigmoid')
%                     dSigma_dA = x_model.*(1 - x_model);
%                     two_diff_times_dSigma_dA = 2*diff.*dSigma_dA;
%                     dF_dbx = sum(two_diff_times_dSigma_dA, 2);
%                     dF_dbz = Wx'*dF_dbx;
%                     dF_dWx = two_diff_times_dSigma_dA*z';
%                     dF_dWz = Wx'*two_diff_times_dSigma_dA*this.trainingData';
%                 else
%                     two_diff = 2*diff;
%                     dF_dbx = sum(two_diff, 2);
%                     dF_dbz = Wx'*dF_dbx;
%                     dF_dWx = two_diff*z';
%                     dF_dWz = Wx'*two_diff*this.trainingData';
%                 end
%                 
%                 %Projections first
%                 d_r = [dF_dWz(:)];
%             end
            d_r = NaN;
            
        end
        
        
        
        function this = train(this)
            %params_0 are parameter start values
                        
            trainingDataMean = mean(this.trainingData, 2);
            trainingDataSqMean = 0;
            trainingData = double(this.trainingData);
            N = size(this.trainingData, 2);
            for n = 1:N
                trainingDataSqMean = trainingDataSqMean +...
                    trainingData(:, n)*trainingData(:, n)';
                if(mod(n, 1000) == 0)
                    n
                end
            end
            clear trainingData;
            trainingDataSqMean = trainingDataSqMean/N;
            xT_Xinv = trainingDataMean'/trainingDataSqMean;
              
            dim = size(this.trainingData, 1);
            latentDim = this.latentDim;
            this.Wx = normrnd(1/latentDim, .1, dim, latentDim);
            WxTWxInvWxT = (this.Wx'*this.Wx)\this.Wx';
            this.Wz = normrnd(1/dim, .1, latentDim, dim);
            WzxMean = this.Wz*trainingDataMean;
            this.bz = zeros(latentDim, 1);
            this.bx = zeros(dim, 1);
            
            res = this.residual(this.bx, this.bz, this.Wx, this.Wz)
            criterion = inf;
            converged = false;
            tol = 1e-3;
            iter = 0;
            tic
            while(~converged)
                this.bx = trainingDataMean - this.Wx*this.bz - this.Wx*WzxMean;
                this.bz = WxTWxInvWxT*(trainingDataMean - this.bx) - WzxMean;
                
                %Wx
                XWzT = trainingDataSqMean*this.Wz';
                bzWzxMeanT = this.bz*WzxMean';
                WxSystem = this.Wz*XWzT + bzWzxMeanT +...
                    bzWzxMeanT' + this.bz*this.bz';
                Wxrhs = XWzT + trainingDataMean*this.bz' -...
                    this.bx*WzxMean' - this.bx*this.bz';
                this.Wx = Wxrhs/WxSystem;
                WxTWxInvWxT = (this.Wx'*this.Wx)\this.Wx';
                
                %Wz
                this.Wz = WxTWxInvWxT -...
                    (this.bz + WxTWxInvWxT*this.bx)*xT_Xinv;
                WzxMean = this.Wz*trainingDataMean;
                
                
                if(mod(iter, 1000) == 0)
                    res_old = res;
                    res = this.residual(this.bx, this.bz, this.Wx, this.Wz)
                    criterion = abs(res - res_old)
                    iter
                    toc
                    tic
                end
                
                if(criterion < tol)
                    converged = true;
                end
                iter = iter + 1;
            end
        end
        
        
        
        function [decodedData] = decode(this, encodedData)
            decodedData = this.Wx*encodedData + this.bx;
        end
        
        
        
        function [encodedData] = encode(this, originalData)
            encodedData = this.Wz*originalData + this.bz;
        end
        
        
        
        function err = reconstructionErr(this, decodedData, trueData)
            binDecodedData = decodedData > .5;
            %Gives fraction of falsely reconstructed pixels
            err = sum(sum(abs(trueData - binDecodedData)))/numel(trueData);
        end
    end
    
end

