classdef StochasticOptimization
    %Class for optimization with noisy gradients
    
    properties
        x = 0;                           %current best X
        gradient                         %current estimated gradient
        gradientHandle                   %handle to gradient function
        momentum                         %current momentum (for e.g. adam update rule)
        steps = 0;                       %Number of performed update steps
        stepOffset = 10000;                %Robbins-Monro step offset
        stepWidth = 1e-3;                %step width parameter
        
        uncenteredXVariance = 0;         %for adam only
        
        updateRule = 'adam';             %Update heuristic
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %stochastic optimization update parameters (different for each heuristic)
        %adam parameters
        beta1 = .7;                     %the higher, the more important is momentum 
        beta2 = .8;                    %curvature parameter
        epsilon = 1e-8;                  %curvature stabilization parameter
        
        debug = false;                    %run debug mode
        
        %Convergence criteria
        maxCompTime = 30;               %maximum computation time in s
        maxIterations = 10000;           %max number of gradient ascent steps
    end
    
    
    
    
    
    methods
        function SOobj = StochasticOptimization(updateRule)
            %constructor
            SOobj.updateRule = updateRule;
        end
        
        
        
        
        function SOobj = update(SOobj)
            %Perform stochastic maximization step
            if strcmp(SOobj.updateRule, 'adam')
                
                if SOobj.steps == 0
                    %careful first iteration
                    SOobj.momentum = 1e-6*SOobj.gradient;
                else
                    SOobj.momentum = SOobj.beta1*SOobj.momentum + (1 - SOobj.beta1)*SOobj.gradient;
                end
                SOobj.uncenteredXVariance = SOobj.beta2*SOobj.uncenteredXVariance...
                    + (1 - SOobj.beta2)*SOobj.gradient.^2;
                
                %Optimization update
                SOobj.x = SOobj.x + (SOobj.stepWidth*SOobj.stepOffset/(SOobj.stepOffset + SOobj.steps)).*...
                    (1./(sqrt(SOobj.uncenteredXVariance) + SOobj.epsilon)).*SOobj.momentum;
                
            elseif strcmp(SOobj.updateRule, 'robbinsMonro')
                delta = ((SOobj.stepWidth*SOobj.stepOffset)/(SOobj.stepOffset + SOobj.steps)).*SOobj.gradient;
                nDelta = norm(delta);
                stabilityFactor = 2;
                if(nDelta > stabilityFactor*norm(SOobj.x))
                    delta = (stabilityFactor/nDelta)*delta;
                end
%                 x = SOobj.x
                SOobj.x = SOobj.x + delta;
            else
                error('Unknown update heuristic for stochastic optimization')
            end
            SOobj.steps = SOobj.steps + 1;
            
        end
        
        
        
        
        function SOobj = converge(SOobj)
            %Run stochastic optimization till convergence criterion is met
            
            if SOobj.debug
                f = figure;
                subplot(1,2,1)
                title('Variational parameters')
                xlabel('iteration')
                subplot(1,2,2)
                title('Gradient')
            end
            
            converged = false;
            SOobj.steps = 0;
            tic;
            while ~converged
                SOobj.gradient = SOobj.gradientHandle(SOobj.x);
                SOobj = SOobj.update;
                
                if SOobj.debug
                    if(mod(SOobj.steps, 1) == 0)
                        figure(f)
                        subplot(1,3,1)
                        hold on;
                        plot(SOobj.steps, SOobj.x, 'xb')
                        axis tight
                        subplot(1,3,2)
                        hold on;
                        plot(SOobj.steps, SOobj.gradient, 'xb')
                        axis tight
                        subplot(1,3,3)
                        hold on;
                        free_mem = java.lang.Runtime.getRuntime.freeMemory
                        plot(SOobj.steps, free_mem, 'xb')
                        drawnow
                        norm_gradient = norm(SOobj.gradient)
                        norm_momentum = norm(SOobj.momentum)
                    end
                end
                
                compTime = toc;
                if SOobj.steps > SOobj.maxIterations
                    converged = true;
                    disp('Converged because max number of iterations exceeded')
                elseif compTime > SOobj.maxCompTime
                    converged = true;
                    disp('Converged because max computation time exceeded')
                end
            end
        end
    end
    
end

