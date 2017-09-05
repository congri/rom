function [log_q, d_log_q, Tc] = log_q_i(Xi, Tf_i_minus_mu, theta_cf, theta_c,...
    Phi,  domainc, condTransOpts)

%Xi must be a column vector
if size(Xi, 2) > 1
    Xi = Xi';
end

[lg_p_c, d_lg_p_c] = log_p_c(Xi, Phi, theta_c);
[lg_p_cf, d_lg_p_cf, Tc] = log_p_cf(Tf_i_minus_mu, domainc, Xi, theta_cf, condTransOpts);

log_q = lg_p_cf + lg_p_c;
d_log_q = d_lg_p_c + d_lg_p_cf;

%Finite difference gradient check
FDcheck = false;
if FDcheck
    disp('Gradient check log q_i')
    conductivity = conductivityBackTransform(Xi(1:domainc.nEl), condTransOpts);
    d = 1e-5;
    if domainc.useConvection
        latentDim = 3*domainc.nEl;
    else
        latentDim = domainc.nEl;
    end
    gradFD = zeros(latentDim, 1);
    for i = 1:latentDim
        dXi = zeros(latentDim, 1);
        dXi(i) = d;
        conductivityFD = conductivity + conductivity.*dXi(1:domainc.nEl);
        
        [lg_p_c, ~] = log_p_c(Xi + dXi, Phi, theta_c);
        [lg_p_cf, ~] = log_p_cf(Tf_i_minus_mu, domainc, Xi + dXi, theta_cf, condTransOpts);
        
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

