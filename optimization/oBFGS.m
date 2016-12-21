function [theta] = oBFGS(meanGradFun, samplesFun, RMstepWidth, RMoff, c, lambda, epsilon, thetaInit)
%Stochastic BFGS, see Schraudolph, Yu, Guenter
debug = true;  %set true for debug output

t = 0;
theta = thetaInit
samples = samplesFun(theta);
grad = meanGradFun(samples, theta)';
dim = length(grad);
I = eye(dim);
B = epsilon*eye(dim);

%Mean gradient over the last k iterations as convergence criterion
k = 5;
maxMeanGrad = 1e-5;
gradArray = zeros(dim, k);
gradArray(:, 1) = grad;

converged = false;
while(~converged)
   
    %RM step size
    eta = (RMstepWidth*RMoff)/(RMoff + t);
    p = - B*grad;
    s = (eta/c)*p;
    theta = (theta + s');
    
    %update gradient
    grad_old = grad;
    theta
    grad = meanGradFun(samples, theta)';
    y = grad - grad_old + lambda*s;
    sy = s'*y;     %ensure that B is pos def with abs?
    syMat = s*y';
    if t == 0
        B = ((sy)/(y'*y))*I;
    end
    rho = 1/(sy);
    
    %Update inverse Hessian
    B = (I - rho*syMat)*B*(I - rho*syMat') + c*rho*(s*s');
    t = t + 1;
    
    %Mean Gradient over the last k iterations
    if t == 1
        mean_k_gradient = .5*(grad + grad_old);
        gradArray(:, 2) = grad;
        grad_s = grad_old;
    elseif t < k
        mean_k_gradient = (t/(t + 1))*mean_k_gradient + (1/(t + 1))*grad;
        gradArray(:, t + 1) = grad;
    else
        mean_k_gradient = mean_k_gradient + (1/k)*(grad - grad_s);
        gradArrayTemp = [gradArray, grad];
        gradArray = gradArrayTemp(:, 2:end);
        grad_s = gradArray(:, 1);
    end
    
    if debug
        grad
        eta
        p
        s
        theta
        y
        sy
        rho
        B
        eigB = eig(B)
        subplot(1,2,1)
        plot(t, theta, 'xr', 'linewidth', 3)
        hold on
        subplot(1,2,2)
        plot(t, mean_k_gradient, 'xr', 'linewidth', 3)
        hold on
        drawnow
    end
    
    
    %Converged?
    mean_k_gradient
    if norm(mean_k_gradient) < maxMeanGrad
        converged = true;
    else
        %resample and go on
        samples = samplesFun(theta);
    end
       
end








end