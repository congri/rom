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
                
                %It is more efficient to repeat the variational approximation a few times,
                %also as this can be run in parallel
                ticBytes(gcp)
                parfor n = 1:N
                    for rep = 1:4
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
                        %                     A = w'*zzT{n};
                        
                        for i = 1:dim
                            %                         xiCell{n}(i) = sqrt(A(i, :)*w(:, i) + twobw(i, :)*mu{n} + bSq(i));
                            xiCell{n}(i) = 2*dataMinus05{n}(i)*(w(:, i)'*mu{n} + b(i));
                        end
                        lambdaCell{n} = lambda(xiCell{n});
                        lambdaCellTimes2{n} = 2*lambdaCell{n};  %for efficiency
                    end
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
            this.xi = cell2mat(xiCell);
            this.mu = cell2mat(mu);
            this.C = C;
            
        end
        
        
        
        function [decodedData, contDecodedData] = decode(this, mu)
            
            if nargin < 2
                mu = this.mu;
                disp('Decode compressed training data...')
            end
            decodedData = heaviside(this.w'*mu + this.b');
            
            %Probability for x_i = 1
            contDecodedData = sigmoid(this.w'*mu + this.b');
        end
        
        
        
        function [mu, C] = encode(this, x)
            
            Nenc = size(x, 2);
            dim = size(x, 1);
            
            %Initialization
            ldim = this.latentDim;
            I = eye(ldim);
            C{1} = I;
            C = repmat(C, 1, Nenc);   %Variational Gaussian mean and cov
            mu{1} = zeros(ldim);  %Column vector
            mu = repmat(mu, 1, Nenc);
            xi = zeros(size(x));   %Variational params, see Tipping
            xiCell = mat2cell(xi, dim, ones(1, Nenc));
            lbda = lambda(xi);
            lambdaCell = mat2cell(lbda, dim, ones(1, Nenc));
            for n = 1:Nenc
                lambdaCellTimes2{n} = 2*lambdaCell{n};
            end
            twobw = zeros(dim, ldim);
            dataMinus05 = x - .5;
            dataMinus05 = mat2cell(dataMinus05, dim, ones(1, Nenc));
            b = this.b;
            w = this.w;
            
            bSq = b.^2;
            for i = 1:dim
                twobw(i, :) = 2*b(i)*w(:, i)';
            end
            
            addpath('./computation');
            ppool = parPoolInit(Nenc);
            converged = false;
            iter = 1;
            maxIter = 25;
            while(~converged)
                parfor n = 1:Nenc
                    mat = 0;
                    vec = 0;
                    for i = 1:dim
                        mat = mat + lambdaCell{n}(i)*(w(:, i)*w(:, i)');
                        vec = vec + (dataMinus05{n}(i) + lambdaCellTimes2{n}(i)*b(i))*w(:, i);
                    end
                    C{n} = inv(I - 2*mat);
                    mu{n} = C{n}*vec;
                    
                    for i = 1:dim
                        xiCell{n}(i) = 2*dataMinus05{n}(i)*(w(:, i)'*mu{n} + b(i));
                    end
                    lambdaCell{n} = lambda(xiCell{n});
                    lambdaCellTimes2{n} = 2*lambdaCell{n};  %for efficiency
                end
                
                if(iter > maxIter)
                    mu = cell2mat(mu);
                    converged = true;
                else
                    iter = iter + 1;
                end
            end
        end
        
        
        
        function err = reconstructionErr(this, decodedData, trueData)
            %Gives fraction of falsely reconstructed pixels
            err = sum(sum(abs(trueData - decodedData)))/numel(trueData);
        end
        
        
    end
    
end

