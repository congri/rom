function [log_p, d_log_p, data] = log_p_c(Xq, Phi, theta_c)
%Probabilistic mapping from fine to coarse heat conductivity
%   Xq:         Effective log conductivity vector
%   Phi:        Design Matrix
%   theta:      basis function coefficients
%   sigma:      noise
%   nFine:      Number of fine elements
%   nCoarse:    Number of coarse elements


if(theta_c.useNeuralNet)
    %theta_c.theta now stores the neural net; Phi stores the microstructure i for q_i
    if isreal(theta_c.theta)
%         theta_c.theta
%         warning('theta_c is a number in neural network mode. First iteration?')
        %THIS SHOULD ONLY HAPPEN IN THE VERY FIRST ITERATION!!!
        mu = 0;
    else
        mu = double(predict(theta_c.theta, Phi));
    end
else
    mu  = Phi*theta_c.theta;    %mean
end

%ignore constant prefactor
% log_p = - size(Xq, 1)*log(sigma) - (1/(2*sigma^2))*(Xq - mu)'*(Xq - mu);
%Diagonal covariance matrix Sigma
log_p = - .5*sum(log(diag(theta_c.Sigma))) - .5*(Xq - mu)'*theta_c.SigmaInv*(Xq - mu);

if nargout > 1
    d_log_p = theta_c.SigmaInv*(mu - Xq);
    
    %Finite difference gradient check
    FDcheck = false;
    if FDcheck
        disp('Gradient check log p_c')
        d = 1e-5;
        d_log_pFD = 0*Xq;
        for i = 1:size(Xq, 1)
            dXq = 0*Xq;
            dXq(i) = d;
            d_log_pFD(i) = (-.5*(Xq + dXq - mu)'*(theta_c.SigmaInv*(Xq + dXq - mu)) - log_p)/d;
        end 
%         d_log_pFD
%         d_log_p
        relGrad = d_log_pFD./d_log_p
    end
end

%dummy
if nargout > 2
    data = 0;
end
    
end

