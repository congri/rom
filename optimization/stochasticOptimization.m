function [Xfinal, steps] = stochasticOptimization(Xinit, gradFunc, sampleFunc, optParams)
%Gradient based stochastic optimization with different update heuristics
debug = false;

X = Xinit;
if strcmp(optParams.optType, 'ASGD')
    Xmean = X;
    XArray = zeros(optParams.XWindow, numel(X));
    XArray(1, :) = X;
elseif (strcmp(optParams.optType, 'adaGrad'))
    gradSq = zeros(1, length(X));
elseif strcmp(optParams.optType, 'adaDelta')
    gradSq = zeros(1, length(X));
    deltaSq = ones(1, length(X));
elseif strcmp(optParams.optType, 'rmsProp')
    gradSq = zeros(1, length(X));
elseif strcmp(optParams.optType, 'adam')
    momentum = zeros(1, length(X));
    uncenteredXVariance = zeros(1, length(X));
end

converged = false;
steps = 0;

if strcmp(optParams.optType, 'oBFGS')
    %UNSTABLE VERSION!!!
    [Xfinal] = oBFGS(gradFunc, sampleFunc, optParams.stepWidth, optParams.offset, .1, 1, 1e-10, X)
else
    while ~converged
        
        samples = sampleFunc(X);
        [grad, gradErr] = gradFunc(samples, X);
        
        Xold = X;
        
        if strcmp(optParams.optType, 'SGD')
            %Simple stochastic gradient descent (ascent here)
            X = X + ((optParams.stepWidth*optParams.offset)/(optParams.offset + steps))*grad;
            
        elseif strcmp(optParams.optType, 'ASGD')
            %Averaged stochastic gradient descent
            
            X = X + ((optParams.stepWidth*optParams.offset)/(optParams.offset + steps))*grad;
            
            %Average dependent variable over last iterations
            if(steps < optParams.XWindow - 1)
                Xmean = ((steps + 1)/(steps + 2))*Xmean + (1/(steps + 2))*X;
                XArray(steps + 2, :) = X;
            else
                XArrayTemp = [XArray; X];
                XArray = XArrayTemp(2:end, :);
                Xmean = mean(XArray);
            end
            X = Xmean;
            
            
        elseif strcmp(optParams.optType, 'adaGrad')
            
            gradSq = gradSq + grad.^2;    %AdaGrad
            rho = ((optParams.stepWidth*optParams.offset)/(optParams.offset + steps))*(gradSq + 1e-8).^(-.5);
            
            %Optimization update
            X = X + rho.*grad;
            
        elseif strcmp(optParams.optType, 'adaDelta')
            
            gradSq = optParams.decayParam*gradSq + (1 - optParams.decayParam)*grad.^2;
            Delta = sqrt(deltaSq./(gradSq + 1e-8)).*grad;
            
            %Optimization update
            X = X + ((optParams.stepWidth*optParams.offset)/(optParams.offset + steps))*Delta;
            deltaSq = optParams.decayParam*deltaSq + (1 - optParams.decayParam)*Delta.^2;
            
        elseif strcmp(optParams.optType, 'rmsProp')
            
            gradSq = optParams.decayParam*gradSq + (1 - optParams.decayParam)*grad.^2;
            X = X + (optParams.stepWidth/(optParams.offset + steps))*...
                (1./sqrt(gradSq + 1e-8)).*grad;
        
        elseif strcmp(optParams.optType, 'adam')
            
            momentum = optParams.adam.beta1*momentum + (1 - optParams.adam.beta1)*grad;
            uncenteredXVariance = optParams.adam.beta2*uncenteredXVariance...
                + (1 - optParams.adam.beta2)*grad.^2;
            
            %Optimization update
            X = X + (optParams.stepWidth*optParams.offset/(optParams.offset + steps))*...
                (1./(sqrt(uncenteredXVariance) + 1e-8)).*momentum;
        end
        
        gradient_norm = norm(grad);
        if(steps == 0)
            gradient_norm_mean = gradient_norm;
            gradient_mean = grad;
            grad_normArray = zeros(1, optParams.gradNormWindow);
            gradArray = zeros(optParams.gradNormWindow, numel(grad));
            grad_normArray(1) = gradient_norm;
            gradArray(1, :) = gradient_mean;
        elseif(steps < optParams.gradNormWindow)
            gradient_norm_mean = (steps/(steps + 1))*gradient_norm_mean + (1/(steps + 1))*gradient_norm;
            gradient_mean = (steps/(steps + 1))*gradient_mean + (1/(steps + 1))*grad;
            grad_normArray(steps + 1) = gradient_norm;
            gradArray(steps + 1, :) = grad;
        else
            grad_normArrayTemp = [grad_normArray, gradient_norm];
            grad_normArray = grad_normArrayTemp(2:end);
            gradient_norm_mean = mean(grad_normArray);
            gradArrayTemp = [gradArray; grad];
            gradArray = gradArrayTemp(2:end, :);
            gradient_mean = mean(gradArray);
        end
        
        if (debug || steps > 500)
            X
            grad
            gradient_norm
            gradient_norm_mean
            gradient_mean
            nmg = norm(gradient_mean)
            relGradErr = gradErr./abs(grad)
            if debug
                %Plot X and gradient
                subplot(1,3,1)
                plot(steps, X, 'rx', 'linewidth', 1)
                axis square
                hold on
                subplot(1,3,2)
                plot(steps, grad, 'bx', 'linewidth', 1)
                axis square
                hold on
                subplot(1,3,3)
                plot(steps, gradient_norm_mean, 'kx', 'linewidth', 3)
                axis square
                hold on
                drawnow
            end
        end
        
        %converged?
        if((steps + 1 > optParams.gradNormWindow))
            if((norm(Xold - X)/norm(X) < optParams.relXtol))
                converged = true;
                Xfinal = X;
                disp('Converged because relXtol < threshold')
            elseif(gradient_norm_mean < optParams.gradNormTol)
                converged = true;
                Xfinal = X;
                disp('Converged because mean norm of grad over last k iterations below threshold')
            elseif(norm(gradient_mean) < optParams.meanGradNormTol)
                converged = true;
                Xfinal = X;
                disp('Converged because norm of mean grad over last k iterations below threshold')
            elseif(steps + 1 > optParams.maxIterations)
                converged = true;
                Xfinal = X;
                disp('Converged because max number of iterations exceeded')
            end
        end
        steps = steps + 1;
    end
end


end

