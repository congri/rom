function [f, d_f] = gaussFiltObjective(precision, designMatrix,...
    distanceMatrix, lambdak, XMeanArray, theta_c, SigmaInv)
%Objective function to optimize parametric Gaussian linear filter

f = 0;
d_f = 0;
for n = 1:numel(designMatrix)
    %Assume that adaptive Gaussian linear filter corresponds to last column
    d_Phi_n = 0*designMatrix{1};
    for k = 1:size(designMatrix{n}, 1)
        designMatrix{n}(k, end) = sum(sum(lambdak{n, k}.*...
            exp(precision*distanceMatrix)));
        d_Phi_n(k, end) = sum(sum(lambdak{n, k}.*distanceMatrix.*...
            exp(precision*distanceMatrix)));
    end
    f = f - (XMeanArray(:, n) - .5*designMatrix{n}*theta_c)'*...
        SigmaInv*designMatrix{n}*theta_c;
    d_f = d_f - (XMeanArray(:, n) - designMatrix{n}*theta_c)'*...
        SigmaInv*d_Phi_n*theta_c;
end
end

