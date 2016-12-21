%main parameter file for 2d coarse-graining
%CHANGE JOBFILE IF YOU CHANGE LINE NUMBERS!
%Number of training data samples
nStart = 1; %start training sample in training data file
nTrain = 4;

%% Initialize coarse domain
genCoarseDomain;
                                                                
%% Generate basis function for p_c
genBasisFunctions;

%% EM params
basisFunctionUpdates = 0;
basisUpdateGap = 80*ceil(nTrain/16);
maxIterations = (basisFunctionUpdates + 1)*basisUpdateGap - 1;

%% Start value of model parameters
%Shape function interpolate in W
theta_cf.W = shapeInterp(domainc, domainf);
%shrink finescale domain object to save memory
domainf = domainf.shrink();
theta_cf.S = (1e-3)*ones(domainf.nNodes, 1);
theta_cf.Sinv = sparse(1:domainf.nNodes, 1:domainf.nNodes, 1./theta_cf.S);
%precomputation to save resources
theta_cf.WTSinv = theta_cf.W'*theta_cf.Sinv;
theta_cf.mu = zeros(domainf.nNodes, 1);
% theta_c.theta = (1/size(phi, 1))*ones(size(phi, 1), 1);
theta_c.theta = 0*ones(nBasis, 1);
% theta_c.theta = 0.5*cos(pi*(1:nBasis)');
d = 10;
% theta_c.theta = 2*d*rand(nBasis, 1) - d;
% theta_c.theta(end) = 1;
% theta_c.theta = 0;
theta_c.sigma = 1e-2;


%what kind of prior for theta_c
theta_prior_type = 'spikeAndSlab';                  %hierarchical_gamma, hierarchical_laplace, laplace, gaussian, spikeAndSlab or none
sigma_prior_type = 'none';
%prior hyperparams; obsolete for no prior
% theta_prior_hyperparam = [0, 1e-10];                   %a and b params for Gamma hyperprior
theta_prior_hyperparam = [.7 1e-6 1e4];
sigma_prior_hyperparam = 1e3;

%% MCMC options
MCMC.method = 'MALA';                                %proposal type: randomWalk, nonlocal or MALA
MCMC.seed = 10;
MCMC.nThermalization = 0;                            %thermalization steps
nSamplesBeginning = [40];
MCMC.nSamples = 40;                                 %number of samples
MCMC.nGap = 40;                                     %decorrelation gap
MCMC.Xi_start = log(.5*loCond + .5*upCond)*ones(domainc.nEl, 1);
%only for random walk
MCMC.MALA.stepWidth = .01;
stepWidth = 2e-0;
MCMC.randomWalk.proposalCov = stepWidth*eye(domainc.nEl);   %random walk proposal covariance
MCMC = repmat(MCMC, nTrain, 1);

%% MCMC options for test chain to find step width
MCMCstepWidth = MCMC;
for i = 1:nTrain
    MCMCstepWidth(i).nSamples = 2;
    MCMCstepWidth(i).nGap = 100;
end

%% Control convergence velocity - take weighted mean of adjacent parameter estimates
mix_sigma = 0;
mix_S = 0;
mix_W = 0;
mix_theta = 0;

%% Variational inference params
dim = domainc.nEl;
VIparams.family = 'diagonalGaussian';
initialParamsArray{1} = [log(.5*loCond + .5*upCond)*ones(1, domainc.nEl) 15*ones(1, domainc.nEl)];
initialParamsArray = repmat(initialParamsArray, nTrain, 1);
VIparams.nSamples = 10;    %Gradient samples per iteration
VIparams.inferenceSamples = 100;
VIparams.optParams.optType = 'adam';
VIparams.optParams.dim = domainc.nEl;
VIparams.optParams.stepWidth = .05;
VIparams.optParams.XWindow = 20;    %Averages dependent variable over last iterations
VIparams.optParams.offset = 10000;  %Robbins-Monro offset
VIparams.optParams.relXtol = 1e-12;
VIparams.optParams.maxIterations = 100;
VIparams.optParams.meanGradNormTol = 30;    %Converged if norm of mean of grad over last k iterations is smaller
VIparams.optParams.gradNormTol = 30;    %Converged if average norm of gradient in last gradNormWindow iterations is below
VIparams.optParams.gradNormWindow = 30;  %gradNormTol
VIparams.optParams.decayParam = .7;   %only works for diagonal Gaussian
VIparams.optParams.adam.beta1 = .9;     %The higher this parameter, the more gradient information from previous steps is retained
VIparams.optParams.adam.beta2 = .999;

%Randomize among data points?
update_qi = 'sequential';    %'randomize' to randomize among data points, 'all' to update all qi's in one E-step



