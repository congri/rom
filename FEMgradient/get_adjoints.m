function [adjoints] = get_adjoints(K, theta_cf, domain, Tf_i_minus_mu_minus_WTc)
%Compute adjoints for gradient computation

%Tc is the full temperature vector here, including essential nodes
d_log_p_cf = theta_cf.WTSinv*Tf_i_minus_mu_minus_WTc;
%Gradient with respect to natural nodes; take out derivatives w.r.t. essential nodes
d_log_p_cf(~isnan(domain.essentialTemperatures)) = [];

adjoints = K\d_log_p_cf;

end

