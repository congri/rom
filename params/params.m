%main parameter file for 2d coarse-graining
%CHANGE JOBFILE IF YOU CHANGE LINE NUMBERS!
%Number of training data samples
nStart = 1; %start training sample in training data file
nTrain = 32;

%Limitation of effective conductivity
condTransOpts.limEffCond = false;
if condTransOpts.limEffCond
    %Upper and lower limit on effective conductivity
    condTransOpts.upperCondLim = upCond;
    condTransOpts.lowerCondLim = loCond;
    condTransOpts.transform = 'logit'; 
end
if ~exist('./data/', 'dir')
    mkdir('./data/');
end
save('./data/conductivityTransformation', 'condTransOpts');

%% Initialize coarse domain
genCoarseDomain;
                                                                
%% Generate basis function for p_c
genBasisFunctions;

%% EM params
basisFunctionUpdates = 0;
basisUpdateGap = 200*ceil(nTrain/16);
maxIterations = (basisFunctionUpdates + 1)*basisUpdateGap - 1;

%% Start value of model parameters
%Shape function interpolate in W
theta_cf.W = shapeInterp(domainc, domainf);
%shrink finescale domain object to save memory
domainf = domainf.shrink();
theta_cf.S = 1*ones(domainf.nNodes, 1);
theta_cf.Sinv = sparse(1:domainf.nNodes, 1:domainf.nNodes, 1./theta_cf.S);
%precomputation to save resources
theta_cf.WTSinv = theta_cf.W'*theta_cf.Sinv;
theta_cf.mu = zeros(domainf.nNodes, 1);
% theta_c.theta = (1/size(phi, 1))*ones(size(phi, 1), 1);
% theta_c.theta = 1*ones(nBasis, 1);
theta_c.theta = 0.1*cos(pi*(1:nBasis)');
% d = .01;
% theta_c.theta = 2*d*rand(nBasis, 1) - d;
% theta_c.theta(end) = 1;
% theta_c.theta = 0;
theta_c.sigma = 1e0;


%what kind of prior for theta_c
theta_prior_type = 'hierarchical_laplace';                  %hierarchical_gamma, hierarchical_laplace, laplace, gaussian, spikeAndSlab or none
sigma_prior_type = 'none';
%prior hyperparams; obsolete for no prior
% theta_prior_hyperparamArray = [0 1e-20];                   %a and b params for Gamma hyperprior
theta_prior_hyperparamArray = [200];
% theta_prior_hyperparam = 10;
sigma_prior_hyperparam = 1e3;

%% MCMC options
MCMC.method = 'MALA';                                %proposal type: randomWalk, nonlocal or MALA
MCMC.seed = 10;
MCMC.nThermalization = 0;                            %thermalization steps
nSamplesBeginning = [40];
MCMC.nSamples = 40;                                 %number of samples
MCMC.nGap = 40;                                     %decorrelation gap

if condTransOpts.limEffCond
    MCMC.Xi_start = conductivityTransform(.5*condTransOpts.upperCondLim +...
        .5*condTransOpts.lowerCondLim, condTransOpts)*ones(domainc.nEl, 1);
else
    MCMC.Xi_start = log(.5*loCond + .5*upCond)*ones(domainc.nEl, 1);
end
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
if condTransOpts.limEffCond
    initialParamsArray{1} = [0*ones(1, domainc.nEl) .1*ones(1, domainc.nEl)];
else
    initialParamsArray{1} = [log(.6*loCond + .4*upCond)*ones(1, domainc.nEl) 5*ones(1, domainc.nEl)];
end
initialParamsArray = repmat(initialParamsArray, nTrain, 1);
VIparams.nSamples = 20;    %Gradient samples per iteration
VIparams.inferenceSamples = 1000;
VIparams.optParams.optType = 'adam';
VIparams.optParams.dim = domainc.nEl;
VIparams.optParams.stepWidth = .1;
VIparams.optParams.XWindow = 20;    %Averages dependent variable over last iterations
VIparams.optParams.offset = 10000;  %Robbins-Monro offset
VIparams.optParams.relXtol = 1e-12;
VIparams.optParams.maxIterations = 50;
VIparams.optParams.meanGradNormTol = 30;    %Converged if norm of mean of grad over last k iterations is smaller
VIparams.optParams.gradNormTol = 30;    %Converged if average norm of gradient in last gradNormWindow iterations is below
VIparams.optParams.gradNormWindow = 30;  %gradNormTol
VIparams.optParams.decayParam = .7;   %only works for diagonal Gaussian
VIparams.optParams.adam.beta1 = .9;     %The higher this parameter, the more gradient information from previous steps is retained
VIparams.optParams.adam.beta2 = .999;

%Randomize among data points?
update_qi = 'sequential';    %'randomize' to randomize among data points, 'all' to update all qi's in one E-step



