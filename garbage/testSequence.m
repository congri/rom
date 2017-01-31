function [theta, sigma2] = testSequence(a, b, c, M, theta_init, sigma2_init, stabilityParam)
%Test sequence for theta_c convergence with Jeffrey's prior

theta = (1/(a + sigma2_init/(theta_init^2 + 2*stabilityParam)))*b;

converged = false;
iter = 0;
while(~converged)
    sigma2 = (1/M)*(c - 2*b*theta + a*theta^2);
    theta = (1/(a + sigma2/(theta^2 + 2*stabilityParam)))*b;
    iter = iter + 1;
    
    if(iter > 1000)
        converged = true;
    end
end


end

