function [out] = meanPoreSize(lambda, phase, conductivities, nElc, nElf, meanOrVar)
%Computes mean pore size density
[p_delta_x, p_delta] = poreSizeDensity(lambda, phase, conductivities, nElc, nElf);

m = p_delta_x'*p_delta;             %mean
if strcmp(meanOrVar, 'mean')
    out = m;
elseif strcmp(meanOrVar, 'var')
    out = (p_delta_x.^2)'*p_delta - m^2;  %variance
else
    error('Mean pore size or variance of pore size?')
end


end

