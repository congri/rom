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
        
        %Convergence criteria
        maxIterations = 100;
    end
    
    methods
        
        
        
        function [this] = train(this)
            %Train autoencoder
            ldim = this.latentDim;  %for parfor
            N = size(this.trainingData, 2);
            dim = size(this.trainingData, 1);
            I = eye(ldim);
            
            rng(1);
            %Initialize
            C{1} = I;
            C = repmat(C, 1, N);   %Variational Gaussian mean and cov
            zzT = C;
            mu{1} = zeros(ldim);  %Column vector
            mu = repmat(mu, 1, N);
            xi = zeros(size(this.trainingData));   %Variational params, see Tipping
            xiCell = mat2cell(xi, dim, ones(1, N));
            lbda = lambda(xi);
            lambdaCell = mat2cell(lbda, dim, ones(1, N));
            for n = 1:N
                lambdaCellTimes2{n} = 2*lambdaCell{n};
            end
            zzT_hat{1} = zeros(ldim + 1, ldim + 1);
            zzT_hat{1}(end) = 1;    %this will never change
            zzT_hat = repmat(zzT_hat, 1, N);
            z_hat{1} = zeros(ldim + 1);
            z_hat = repmat(z_hat, 1, N);
            w = .1*rand(ldim, dim) - .05;
            w_hat = [w; zeros(1, dim)];
            b = .1*rand(1, dim) - .05;
            twobw = zeros(dim, ldim);
            dataMinus05 = this.trainingData - .5;
            dataMinus05 = mat2cell(dataMinus05, dim, ones(1, N));
            
            addpath('./computation');
            ppool = parPoolInit(N);
            
            iter = 0;
            converged = false;
            tic;
            while(~converged)
                
                bSq = b.^2;
                for i = 1:dim
                    twobw(i, :) = 2*b(i)*w(:, i)';
                end
                
                ticBytes(gcp)
                parfor n = 1:N
                    mat = 0;
                    vec = 0;
                    for i = 1:dim
                        mat = mat + lambdaCell{n}(i)*(w(:, i)*w(:, i)');
                        vec = vec + (dataMinus05{n}(i) + lambdaCellTimes2{n}(i)*b(i))*w(:, i);
                    end
                    C{n} = inv(I - 2*mat);
                    mu{n} = C{n}*vec;
                    
                    zzT{n} = C{n} + mu{n}*mu{n}';
                    zzT_hat{n}(1:ldim, 1:ldim) = zzT{n};
                    zzT_hat{n}(end, 1:ldim) = mu{n}';
                    zzT_hat{n}(1:ldim, end) = mu{n};
                    z_hat{n} = [mu{n}; 1];
                    A = w'*zzT{n};
                    
                    for i = 1:dim
                        xiCell{n}(i) = sqrt(A(i, :)*w(:, i) + twobw(i, :)*mu{n} + bSq(i));
                    end
                    lambdaCell{n} = lambda(xiCell{n});
                    lambdaCellTimes2{n} = 2*lambdaCell{n};  %for efficiency
                end
                tocBytes(gcp)
                
                for i = 1:dim
                    mat = 0;
                    vec = 0;
                    for n = 1:N
                        mat = mat + lambdaCellTimes2{n}(i)*zzT_hat{n};
                        vec = vec + dataMinus05{n}(i)*z_hat{n};
                    end
                    w_hat(:, i) = -mat\vec;
                    w(:, i) = w_hat(1:ldim, i);
                    b(i) = w_hat(end, i);
                end
                w_hat1 = w_hat(:, 1)
                hold on;
                plot(iter, w_hat1, 'bx')
                drawnow
                
                if iter > this.maxIterations
                    converged = true;
                end
                iter = iter + 1
                toc
                tic;
            end
            this.w = w;
            this.b = b;
            this.xi = xi;
            this.mu = cell2mat(mu);
            this.C = C;
            [~, ~, err] = this.decode;
            err
            
        end
        
%         function [l] = lambda(this, xi)
%             %l = (.5 - sigmoid(xi))./(2*xi + eps);
%             l = (.5 - 1./(1 + exp(-xi)))./(2*xi);
%             l(xi == 0) = -.125;
%         end
        
        function [decodedData, contDecodedData, err] = decode(this)
            
            decodedData = heaviside(this.w'*this.mu + this.b');
            
            %Probability for x_i = 1
            contDecodedData = sigmoid(this.w'*this.mu + this.b');
            
            err = sum(sum(abs(this.trainingData - decodedData)))/numel(this.trainingData);
            
        end
        
    end
    
end

