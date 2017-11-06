function [varDistParams, x] = efficientStochOpt(x, log_emp_dist, variationalDist, stepWidth, dim)
%Memory efficient stocastic optimization for parfor loop
%Perform stochastic maximization step

debug = false;   %debug mode

updateRule = 'adam';
beta1 = .7;                     %the higher, the more important is momentum
beta2 = .8;                    %curvature parameter
epsilon = 1e-8;                  %curvature stabilization parameter

stepOffset = 10000;                %Robbins-Monro step offset
nSamples = 10;                  %gradient samples per iteration

converged = false;
steps = 0;
maxIterations = 10000;
maxCompTime = 15;
tic;

if strcmp(variationalDist, 'diagonalGauss')
    varDistParams.mu = x(1:dim);
    varDistParams.sigma = exp(-.5*x((dim + 1):end));
elseif strcmp(variationalDist, 'fullRankGauss')
    varDistParams.mu = x(1:dim);
    %lower triangular cholesky factor L
    varDistParams.L = reshape(x((dim + 1):end), dim, dim);
    varDistParams.LT = varDistParams.L';
    varDistParams.LInv = inv(varDistParams.L);
    varDistParams.Sigma = varDistParams.L*varDistParams.LT;
else
    error('Unknown variational distribution')
end

if debug
    f = figure;
end

iter = 0;
while ~converged
        
    gradient = sampleELBOgrad(log_emp_dist, variationalDist, nSamples, varDistParams);
    
    if strcmp(updateRule, 'adam')
        
        if steps == 0
            %careful first iteration
            momentum = 1e-6*gradient;
            uncenteredXVariance = gradient.^2;
        else
            momentum = beta1*momentum + (1 - beta1)*gradient;
        end
        uncenteredXVariance = beta2*uncenteredXVariance...
            + (1 - beta2)*gradient.^2;
        
        %Optimization update
        x = x + (stepWidth*stepOffset/(stepOffset + steps)).*...
            (1./(sqrt(uncenteredXVariance) + epsilon)).*momentum;
        
    elseif strcmp(updateRule, 'robbinsMonro')
        delta = ((stepWidth*stepOffset)/(stepOffset + steps)).*gradient;
        nDelta = norm(delta);
        stabilityFactor = 2;
        if(nDelta > stabilityFactor*norm(x))
            delta = (stabilityFactor/nDelta)*delta;
        end
        x = x + delta;
    else
        error('Unknown update heuristic for stochastic optimization')
    end
    steps = steps + 1;
    
    if strcmp(variationalDist, 'diagonalGauss')
        varDistParams.mu = x(1:dim);
        varDistParams.sigma = exp(-.5*x((dim + 1):end));
    elseif strcmp(variationalDist, 'fullRankGauss')
        varDistParams.mu = x(1:dim);
        %lower triangular cholesky factor L
        varDistParams.L = reshape(x((dim + 1):end), dim, dim);
        varDistParams.LT = varDistParams.L';
        varDistParams.LInv = inv(varDistParams.L);
        varDistParams.Sigma = varDistParams.L*varDistParams.LT;
    else
        error('Unknown variational distribution')
    end
    
    if debug
        figure(f);
        if(mod(iter,1) == 0)
            varDistParams.mu
            varDistParams.sigma
            subplot(1,2,1)
            hold on;
            title('mu')
            plot(iter, varDistParams.mu, 'bx')
            subplot(1,2,2)
            hold on;
            title('sigma')
            plot(iter, varDistParams.sigma, 'rx')
            drawnow
        end
    end
    
    compTime = toc;
    if steps > maxIterations
        converged = true;
        disp('Converged because max number of iterations exceeded')
    elseif compTime > maxCompTime
        converged = true;
        disp('Converged because max computation time exceeded')
    end
    iter = iter + 1;
end

if strcmp(variationalDist, 'diagonalGauss')
    varDistParams.XSqMean = varDistParams.sigma.^2 + varDistParams.mu.^2;
elseif strcmp(variationalDist, 'fullRankGauss')
    varDistParams.XSqMean = diag(varDistParams.Sigma + varDistParams.mu'*varDistParams.mu);
else
    error('Unknown variational distribution')
end

