function [Xopt, LambdaOpt, s2, thetaOpt, s2theta, LambdaThetaOpt] = detOpt_p_cf(dataPoint, nStart, nTrain)
%Deterministic optimization of log(p_cf) to check capabilities of model

addpath('./heatFEM')
addpath('./rom')


%Conductivity transformation options
condTransOpts.anisotropy = false;
%Upper and lower limit on effective conductivity
condTransOpts.upperCondLim = 1e10;
condTransOpts.lowerCondLim = .001;
condTransOpts.transform = 'log_lower_bound';

%Load finescale data, including domainf
loadTrainingData;
%Initialize coarse domain
genCoarseDomain;


theta_cf.S = 1;
theta_cf.Sinv = 1;
theta_cf.W = shapeInterp(domainc, domainf);
theta_cf.WTSinv = theta_cf.W'*theta_cf.Sinv;

Tf = Tffile.Tf(:, dataPoint);

objFun = @(X) objective(X, Tf, domainc, condTransOpts, theta_cf);


options = optimoptions(@fminunc,'Display','iter', 'SpecifyObjectiveGradient', true);
Xinit = ones(domainc.nEl, 1);
[Xopt, fval] = fminunc(objFun, Xinit, options);
LambdaOpt = conductivityBackTransform(Xopt, condTransOpts);

%s2 is the squared distance of truth to optimal coarse averaged over all nodes
s2 = (2*fval)/domainf.nNodes;


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

