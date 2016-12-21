function [phi, PhiArray, sumPhiSq] = updateBasisFunction(XMean, x, theta, PhiArray, nFine, nCoarse, phi)
%Still needs to be generalized for 2d!!!
disp('Updating basis functions phi in p_c...')

%For generalized means
%         [thetaTildeOpt, zOpt] = optNewPhi(XMean, x, theta, PhiArray, nFine, nCoarse);
%         zOptVec = [zOptVec zOpt]
%         if zOpt >= 30
%             phi{end + 1, 1} = @(x) log(max(x));
%         elseif zOpt <= -30
%             phi{end + 1, 1} = @(x) log(min(x));
%         else
%             phi{end + 1, 1} = @(x) (1/zOpt)*log((1/FperC)*sum(x.^zOpt));
%         end
%         theta_c.theta(end + 1, 1) = thetaTildeOpt;

%For predefined basis functions
if numel(phi) == 1
    phi{2, 1} = phi_2;
    theta_c.theta = [theta; 0];
elseif numel(phi) == 2
    phi{3, 1} = phi_1;
    theta_c.theta = [theta; 0];
elseif numel(phi) == 3
    phi{4, 1} = phi_4;
    theta_c.theta = [theta; 0];
else
    error('Which basis function to add?')
end

% Compute and store design matrix for each data point
PhiArray = zeros(domainc.nEl, numel(phi), nTrain);
for i = 1:size(cond, 2)
    PhiArray(:,:,i) = designMatrix(phi, cond, domainf, domainc);
end
% Compute inverse of sum_i Phi^T(x_i)^Phi(x_i)
sumPhiSq = zeros(numel(phi), numel(phi));
for i = 1:nTrain
    sumPhiSq = sumPhiSq + PhiArray(:,:,i)'*PhiArray(:,:,i);
end

end

