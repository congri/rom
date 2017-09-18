function [obj, gradObj] = objective(Xi, Tf, domainc, condTransOpts, theta_cf)
%Objective function for deterministic optimization of log_p_cf

dim1 = false;
if dim1
    %Coarse discretization of random field is of dim 1
    X = Xi*ones(domainc.nEl, 1);
else
    X = Xi;
end
[lg_pcf, d_lg_pcf] = log_p_cf(Tf, domainc, Xi, theta_cf, condTransOpts);
obj = -2*lg_pcf;
gradObj = -2*d_lg_pcf;

if dim1
   gradObj = sum(gradObj); 
end
end