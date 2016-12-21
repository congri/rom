function [curr_x, grad, nIter] = BFGSMaximization(objective, startValue, options)
%Gradient-based maximization using BFGS, where there is no need to provide the objective function
%output (only gradient is sufficient). See Wikipedia on BFGS.
%   Input:  objective:              handle to objective function, output: [grad, obj]

%notation
curr_x = startValue;                %current x
Hinv = - eye(length(curr_x));       %Inverse Hessian
stepSize = options.initialStepSize;
stepSizeRM = .1;
converged = false;
nIter = 0;

if(options.provide_objective)
    
else
    %output of objective function not provided
    while(~converged)
        grad = objective(curr_x);
        nIter = nIter + 1;
        direction = - Hinv*grad;
        step = stepSize*direction;      %not only needed for optimization, but also Hessian computation
        %check for negative definiteness
%         if(~all(all(isfinite(Hinv))) || any(eig(Hinv) >= 0))
        if(true)
            warning('Hessian not negative definite, do Robbins-Monro instead')
            RMoff = 100;
            RMstep = stepSizeRM*RMoff/(RMoff + nIter)
            stepRM = RMstep*norm(curr_x)*(grad/norm(grad));
%             %upper cutoff
%             if(norm(stepRM) > 3*norm(curr_x))
%                stepRM = 3*norm(curr_x)*(stepRM/norm(stepRM)); 
%             end
            curr_x = curr_x + stepRM;
            grad_old = grad;
            grad = objective(curr_x);
            if any((sign(grad) == sign(grad_old)) == 0)
                %if sign of any gradient component changes, decrease step size (heuristic)
                stepSizeRM = .9*stepSizeRM;
            else
                %increase step size
                stepSizeRM = 1.1*stepSizeRM;
            end
        else
%             %upper cutoff
%             if(norm(step) > 3*norm(curr_x))
%                step = 3*norm(curr_x)*(step/norm(step)); 
%             end
            curr_x = curr_x + step;
            grad_old = grad;
            grad = objective(curr_x);
            if any((sign(grad) == sign(grad_old)) == 0)
                %if sign of any gradient component changes, decrease step size (heuristic)
                stepSize = .7*stepSize;
            else
                %increase step size
                stepSize = 1.3*stepSize;
            end
        end
        
        diffGrad = grad - grad_old;
        stepDiffGrad = step'*diffGrad;
        HinvDiffGrad = Hinv*diffGrad;
        %Update inverse Hessian
        Hinv = Hinv + ((stepDiffGrad + diffGrad'*HinvDiffGrad)/(stepDiffGrad^2))*(step*step')...
            - (HinvDiffGrad*step' + (HinvDiffGrad*step')')/(stepDiffGrad);
        if(~all(all(isfinite(Hinv))))
            Hinv
            stepDiffGrad
            grad
            grad_old
            HinvDiffGrad
            step
            warning('Non-finite inverse Hessian Hinv - set it to -I')
            Hinv = - eye(length(curr_x));
        end
        
        
        if(options.debug)
            plot(nIter*ones(1, length(grad)), grad, 'rx', 'markersize', 8, 'linestyle', 'none', 'linewidth', 2)
            xlim([0 nIter])
            hold on;
            drawnow;
%             step
%             stepSize
            direction
            grad
            Hinv
            curr_x
            if(mod(nIter, 1000) == 0)
                pause;
            end
        end
        
        %convergence check
        if(norm(grad)/norm(curr_x) < options.gradTol)
            converged = true;
        end
    end
    
    
end

