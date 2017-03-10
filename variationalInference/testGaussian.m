function [log_p, d_log_p] = testGaussian(x, covarVec, muVec)

dim = length(muVec);
trueCovarInvDiagMat = sparse(1:dim, 1:dim, 1./covarVec);
log_p = -.5*dim*log(2*pi) - .5*sum(log(covarVec)) - .5*(x - muVec)*trueCovarInvDiagMat*(x - muVec)';
d_log_p = trueCovarInvDiagMat*(muVec - x)';

end

