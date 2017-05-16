function [log_p, d_log_p] = testGaussian(x, precision, muVec)

dim = length(muVec);
log_p = -.5*dim*log(2*pi) + .5*logdet(precision) - .5*(x - muVec)*precision*(x - muVec)';
d_log_p = precision*(muVec - x)';

end

