classdef BinaryAutoencoder
    %See Tipping: Probabilistic visualization of high dimensional binary data
    properties
        trainingData              %Binary (image) data, D x N
        latentDim = 4;
        
        %parameters
        w
        b
        xi
        mu
        C
    end
    
    methods
        
        
        
        function [this] = train(this)
            %Train autoencoder
            N = size(this.trainingData, 2);
            dim = size(this.trainingData, 1);
            I = eye(this.latentDim);
            
            %Initialize
            C = repmat(I, 1, 1, N);   %Variational Gaussian mean and cov
            zzT = C;
            mu = zeros(this.latentDim, N);  %Column vector
            xi = zeros(size(this.trainingData));   %Variational params, see Tipping
            xiSq = xi.^2;
            lambda = this.lambda(xi);
            zzT_hat = zeros(this.latentDim + 1, this.latentDim + 1, N);
            z_hat = zeros(this.latentDim + 1, N);
            w = ones(this.latentDim, dim)/this.latentDim;
            w_hat = zeros(this.latentDim + 1, dim);
            b = zeros(1, dim);
            twobw = zeros(dim, this.latentDim);
            
            iter = 0;
            converged = false;
            while(~converged)
                
                bSq = b.^2;
                for i = 1:dim
                    twobw(i, :) = 2*b(i)*w(:, i)';
                end
                for n = 1:N
                    mat = 0;
                    vec = 0;
                    dataMinus05 = this.trainingData - .5;
                    lambdaTimes2 = 2*lambda;
                    for i = 1:dim
                        mat = mat + lambda(i, n)*(w(:, i)*w(:, i)');
                        vec = vec + (dataMinus05(i, n) + lambdaTimes2(i, n)*b(i))*w(:, i);
                    end
                    C(:, :, n) = inv(I - 2*mat);
                    mu(:, n) = C(:, :, n)*vec;
                    
                    zzT(:, :, n) = C(:, :, n) + mu(:, n)*mu(:, n)';
                    zzT_hat(1:this.latentDim, 1:this.latentDim, n) = zzT(:, :, n);
                    zzT_hat(end, 1:this.latentDim, n) = mu(:, n)';
                    zzT_hat(1:this.latentDim, end, n) = mu(:, n);
                    zzT_hat(end, end, n) = 1;
                    z_hat(:, n) = [mu(:, n); 1];
                    A = w'*zzT(:, :, n);
                    
                    for i = 1:dim
                        xiSq(i, n) = A(i, :)*w(:, i) + twobw(i, :)*mu(:, n) + bSq(i);
                    end
                end
                xi = sqrt(xiSq);
                lambda = this.lambda(xi);
               
                
                for i = 1:dim
                    mat = 0;
                    vec = 0;
                    for n = 1:N
                        mat = mat + 2*lambda(i, n)*zzT_hat(:, :, n);
                        vec = vec + (this.trainingData(i, n) - .5)*z_hat(:, n);
                    end
                    w_hat(:, i) = -mat\vec;
                    w(:, i) = w_hat(1:this.latentDim, i);
                    b(i) = w_hat(end, i);
                end
                w_hat1 = w_hat(:, 1)
                hold on;
                plot(iter, w_hat1, 'bx')
                drawnow
                                
                if iter > 60
                    converged = true;
                end
                iter = iter + 1
            end
            this.w = w;
            this.b = b;
            this.xi = xi;
            this.mu = mu;
            this.C = C;
  
        end
        
        function [l] = lambda(this, xi)
            %l = (.5 - sigmoid(xi))./(2*xi + eps);
            l = (.5 - 1./(1 + exp(-xi)))./(2*xi);
            l(xi == 0) = -.125;
        end
        
        function [decodedData, contDecodedData] = decode(this)
            
            decodedData = heaviside(this.w'*this.mu + this.b');
            
            %Probability for x_i = 1
            contDecodedData = sigmoid(this.w'*this.mu + this.b');
            
        end
        
    end
    
end

