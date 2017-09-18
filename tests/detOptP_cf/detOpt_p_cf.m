function [Xopt, LambdaOpt, s2, thetaOpt, s2theta, LambdaThetaOpt] = detOpt_p_cf(nStart, nTrain)
%Deterministic optimization of log(p_cf) to check capabilities of model

addpath('./heatFEM')
addpath('./rom')
addpath('./params')
addpath('./aux')
addpath('./FEMgradient')
addpath('./featureFunctions')


romObj = ROM_SPDE('train');

%don't change these!
romObj.theta_cf.S = 1;
romObj.theta_cf.Sinv = 1;
romObj.theta_cf.Sinv_vec = ones(romObj.fineScaleDomain.nNodes, 1);
romObj.theta_cf.W = shapeInterp(romObj.coarseScaleDomain, romObj.fineScaleDomain);
romObj.theta_cf.WTSinv = romObj.theta_cf.W'*romObj.theta_cf.Sinv;
romObj.theta_cf.sumLogS = 0;

options = optimoptions(@fminunc,'Display','off', 'SpecifyObjectiveGradient', true);
Xinit = 0*ones(romObj.coarseScaleDomain.nEl, 1);
Xopt = zeros(romObj.coarseScaleDomain.nEl, nTrain);
LambdaOpt = Xopt;
s2 = zeros(1, nTrain);
j = 1;
for i = nStart:(nStart + nTrain -1)
    Tf = romObj.testDataMatfile.Tf(:, i);
    objFun = @(X) objective(X, Tf, romObj.coarseScaleDomain, romObj.conductivityTransformation, romObj.theta_cf);
    [XoptTemp, fvalTemp] = fminunc(objFun, Xinit, options);
    LambdaOptTemp = conductivityBackTransform(XoptTemp, romObj.conductivityTransformation);
    Xopt(:, j) = XoptTemp;
    LambdaOpt(:, j) = LambdaOptTemp;
    
    %s2 is the squared distance of truth to optimal coarse averaged over all nodes
    s2(j) = fvalTemp/romObj.fineScaleDomain.nNodes;
    j = j + 1
end


if nargout > 3
    %% X_k = theta_i*phi_i(x_k)
    %Generate basis function for p_c
    genBasisFunctions;
    
    Phi = DesignMatrix([domainf.nElX domainf.nElY], [domainc.nElX domainc.nElY], phi, Tffile, nStart:(nStart + nTrain - 1));
    Phi = Phi.computeDesignMatrix(domainc.nEl, domainf.nEl, condTransOpts);
    %Normalize design matrices
    % Phi = Phi.standardizeDesignMatrix;
    Phi = Phi.rescaleDesignMatrix;
    
    
    objFun2 = @(theta) thetaObjective(theta, Phi, nStart, nTrain, domainc, Tf, condTransOpts, theta_cf);
    thetaInit = zeros(numel(phi), 1);
    [thetaOpt, fval] = fminunc(objFun2, thetaInit, options);
    s2theta = (2*fval)/(domainf.nNodes*nTrain);
    
    LambdaThetaOpt = zeros(domainc.nEl, nTrain);
    j = 1;
    for i = nStart:(nStart + nTrain - 1)
        LambdaThetaOpt(:, j) = exp(Phi.designMatrices{i}*thetaOpt);
        j = j + 1;
    end
end

end

