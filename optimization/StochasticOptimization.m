classdef StochasticOptimization
    %Class for optimization with noisy gradients
    
    properties
        x = 0;                           %current best X
        gradient                         %current estimated gradient
        momentum                         %current momentum (for e.g. adam update rule)
        steps = 0;                       %Number of performed update steps
        stepOffset = 100;                %Robbins-Monro step offset
        stepWidth = 2e-2;                %step width parameter
        
        uncenteredXVariance = 0;         %for adam only
        
        updateRule = 'adam';             %Update heuristic
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %stochastic optimization update parameters (different for each heuristic)
        %adam parameters
        beta1 = .85;                     %the higher, the more important is momentum 
        beta2 = .999;                    %curvature parameter
        epsilon = 1e-8;                  %curvature stabilization parameter
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
                    SOobj.momentum = .001*SOobj.gradient;
                else
                    SOobj.momentum = SOobj.beta1*SOobj.momentum + (1 - SOobj.beta1)*SOobj.gradient;
                end
                SOobj.uncenteredXVariance = SOobj.beta2*SOobj.uncenteredXVariance...
                    + (1 - SOobj.beta2)*SOobj.gradient.^2;
                
                %Optimization update
                SOobj.x = SOobj.x + (SOobj.stepWidth*SOobj.stepOffset/(SOobj.stepOffset + SOobj.steps))*...
                    (1./(sqrt(SOobj.uncenteredXVariance) + SOobj.epsilon)).*SOobj.momentum;
                
            elseif strcmp(SOobj.updateRule, 'robbinsMonro')
                SOobj.x = SOobj.x + ...
                    ((SOobj.stepWidth*SOobj.stepOffset)/(SOobj.stepOffset + SOobj.steps))*SOobj.gradient;
            else
                error('Unknown update heuristic for stochastic optimization')
            end
            SOobj.steps = SOobj.steps + 1;
            
        end
    end
    
end

