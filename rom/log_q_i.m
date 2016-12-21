function [log_q, d_log_q, Tc] = log_q_i(Xi, Tf_i_minus_mu, theta_cf, theta_c, Phi,  domainc)

%Xi must be a column vector
if size(Xi, 2) > 1
    Xi = Xi';
end

conductivity = logCond2Cond(Xi, 1e-10, 1e10);

[lg_p_c, d_lg_p_c] = log_p_c(Xi, Phi, theta_c.theta, theta_c.sigma);
[lg_p_cf, d_lg_p_cf, Tc] = log_p_cf(Tf_i_minus_mu, domainc, conductivity, theta_cf);

log_q = lg_p_cf + lg_p_c;

d_log_q = d_lg_p_c + d_lg_p_cf;


%Finite difference gradient check
FDcheck = false;
if FDcheck
    disp('Gradient check log q_i')
    d = 1e-5;
    gradFD = zeros(domainc.nEl, 1);
    for i = 1:domainc.nEl
        dXi = zeros(domainc.nEl, 1);
        dXi(i) = d;
        conductivityFD = conductivity + conductivity.*dXi;
        
        [lg_p_c, ~] = log_p_c(Xi + dXi, Phi, theta_c.theta, theta_c.sigma);
        [lg_p_cf, ~] = log_p_cf(Tf_i_minus_mu, domainc, conductivityFD, theta_cf);
        
        log_qFD = lg_p_cf + lg_p_c;
        gradFD(i) = (log_qFD - log_q)/d;
    end
    
    relgrad = gradFD./d_log_q
    if(any(abs(relgrad - 1) > .1))
        %Note: if d_log_q << d_log_p_c, d_log_p_cf, then this might be due to numerical issues, i.e.
        %FD gradient is unprecise
        %for small log q, it is possible that the FD gradient is unprecise
        conductivity
        conductivityFD
        Xi
        XiFD = Xi + dXi
        d_log_q
        d_lg_p_c
        d_lg_p_cf
        pause 
    end
    
end

end

