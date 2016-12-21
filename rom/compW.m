function [W] = compW(Tc_dyadic_mean, rhsTc, Sinv, W, paramIndices, constIndices)
%% Computes the free parameters of the W matrix of the linear mapping in p_cf
% Tc_dyadic_mean:           mean of Tc*Tc^T over MCMC and data samples
% rhsTc:                    (1/N)*sum_i (Tf_i - mu)*<Tc>_qi^T
% mu:                       model parameter mu
% Tf:                       FOM data output; samples in column index
% Sinv:                     model parameter S^(-1)
% W:                        W-matrix from previous step; only needed to find non-param elements in W
% paramIndices:             Indices that correspond to free params in W


%% Collect free parameter coefficients; assemble system matrix
neq = size(paramIndices, 1);
SysMat = zeros(neq);    %system matrix
rhs = zeros(neq, 1);    %system right hand side
for i = 1:neq
    p = paramIndices(i, 1);
    q = paramIndices(i, 2);
    rhs(i) = Sinv(p, :)*rhsTc(:, q);
    %loop through param indices for system matrix
    for j = 1:neq
        a = paramIndices(j, 1);
        b = paramIndices(j, 2);
        SysMat(i, j) = Tc_dyadic_mean(q, b)*Sinv(p, a);
    end

    %loop through constant indices for right hand side
    for j = 1:size(constIndices, 1)
        a = constIndices(j, 1);
        b = constIndices(j, 2);
        rhs(i) = rhs(i) - Tc_dyadic_mean(q, b)*Sinv(p, a)*W(a, b);
    end
end

%% Solve equation system
w = SysMat\rhs;

%% Assemble new W-matrix
for i = 1:neq
    W(paramIndices(i, 1), paramIndices(i, 2)) = w(i);
end

end

