function [p_cf_exp] = sqMisfit(X, condTransOpts, domainc, Tf_i_minus_mu, theta_cf)
%function computing the squared difference between prediction and truth
%   X:  transformed conductivity (row vector)

%transformed conductivity to conductivity
conductivity = conductivityBackTransform(X, condTransOpts)

%Set up conductivity tensors for each element
for j = 1:domainc.nEl
    D(:, :, j) =  conductivity(j)*eye(2);
end

%Solve coarse FEM model
FEMout = heat2d(domainc, D);

Tc = FEMout.Tff';
Tc = Tc(:);

p_cf_exp = (Tf_i_minus_mu - theta_cf.W*Tc).^2;
end

