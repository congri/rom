function [x] = genConductivity(fineCond, N_el)
%Generate conductivities from log-normal

%a single sample of the conductivity y corresponds to a column of y
x = lognrnd(fineCond.mu, fineCond.sigma, N_el, fineCond.nSamples);

end

