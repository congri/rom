function [obj, gradObj] = thetaObjective(theta, Phi, nStart, nTrain, domainc, Tf, condTransOpts, theta_cf)
%Objective function for deterministic optimization for X_k^i = Phi_i(x_k)

j = 1;
obj = 0;
gradObj = 0;
for i = nStart:(nStart + nTrain - 1)
    X = Phi.designMatrices{i}*theta;
    [objTemp, gradObjTemp] = objective(X, Tf, domainc, condTransOpts, theta_cf);
    obj = obj + objTemp;
    gradObj = gradObj + Phi.designMatrices{i}'*gradObjTemp;
    j = j + 1;
end



end

