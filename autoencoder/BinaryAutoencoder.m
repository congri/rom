classdef BinaryAutoencoder
    %See Tipping: Probabilistic visualization of high dimensional binary data
    properties
        trainingData              %Binary (image) data, D x N
        latentDim = 2;
        
        %parameters
        w
        b
        xi
    end
    
    methods
        
        
        
        function [this] = train(this)
            %Train autoencoder
            N = size(this.trainingData, 2);
            dim = size(this.trainingData, 1);
            I = eye(this.latentDim);
            
            %Initialize
            C = repmat(I, 1, 1, N);   %Variational Gaussian mean and cov
            mu = zeros(this.latentDim, N);  %Column vector
            xi = ones(size(this.trainingData));   %Variational params, see Tipping
            zzT = C;    %= <z*z^T>_tilde{p}
            zzT_hat = zeros(this.latentDim + 1, this.latentDim + 1, N);
            z_hat = zeros(this.latentDim + 1, N);
            w = zeros(this.latentDim, dim);
            w_hat = zeros(this.latentDim + 1, dim);
            b = zeros(1, dim);

            iter = 0;
            converged = false;
            while(~converged)

                for n = 1:N
                    zzT(:, :, n) = zzT(:, :, n) + mu(:, n)*mu(:, n)';
                    zzT_hat(1:this.latentDim, 1:this.latentDim, n) = zzT(:, :, n);
                    zzT_hat(end, 1:this.latentDim, n) = mu(:, n)';
                    zzT_hat(1:this.latentDim, end, n) = mu(:, n);
                    zzT_hat(end, end, n) = 1;
                    z_hat(:, n) = [mu(:, n); 1];
                end
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
                
                for n = 1:N
                    mat = 0;
                    vec = 0;
                    for i = 1:dim
                        mat = mat + lambda(i, n)*(w(:, i)*w(:, i)');
                        vec = vec + (this.trainingData(i, n) - .5 + 2*lambda(i, n)*b(i))*w(:, i);
                    end
                    C(:, :, n) = inv(I - 2*mat);
                    mu(:, n) = C(:, :, n)*vec;
                    
                    for i = 1:dim
                        xi(i, n) = sqrt(w(:, i)'*(C(:, :, n) + mu(:, n)*mu(:, n)')*w(:, i) +...
                            2*b(i)*w(:, i)'*mu(:, n) + b(i)^2);
                    end
                end
                
                if iter > 5
                    converged = true;
                end
                iter = iter + 1
            end
            this.w = w;
            this.b = b;
            this.xi = xi;
  
        end
        
        function [l] = lambda(this, xi)
            l = (.5 - sigmoid(xi))./(2*xi);
        end
        
    end
    
end

