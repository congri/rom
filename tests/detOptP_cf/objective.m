function [obj, gradObj] = objective(Xi, Tf, domainc, condTransOpts, theta_cf)
%Objective function for deterministic optimization of log_p_cf

[lg_pcf, d_lg_pcf] = log_p_cf(Tf, domainc, conductivityBackTransform(Xi, condTransOpts), theta_cf, condTransOpts);
obj = -lg_pcf;
gradObj = - d_lg_pcf;
end