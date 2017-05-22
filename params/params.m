%main parameter file for 2d coarse-graining
%CHANGE JOBFILE IF YOU CHANGE LINE NUMBERS!
%Number of training data samples
nStart = 1; %start training sample in training data file
nTrain = 16;

%Anisotropy; do NOT use together with limEffCond
condTransOpts.anisotropy = false;
%Upper and lower limit on effective conductivity
% condTransOpts.upperCondLim = 1.1*upCond;
% condTransOpts.lowerCondLim = .96*loCond;
% condTransOpts.shift = 4;    %controls eff. cond. for all theta_c = 0
condTransOpts.upperCondLim = 1e8;
condTransOpts.lowerCondLim = 1e-8;
%options:
%log: x = log lambda
%log_lower_bound: x = log(lambda - lower_bound), i.e. lambda > lower_bound
%logit: lambda = logit(x), s.t. lowerCondLim < lambda < upperCondLim
%log_cholesky: unique cholesky decomposition for anisotropy
condTransOpts.transform = 'log';
if ~exist('./data/', 'dir')
    mkdir('./data/');
end
save('./data/conductivityTransformation', 'condTransOpts');

%% Initialize coarse domain
genCoarseDomain;
                                                                
%% Generate basis function for p_c
genBasisFunctions;
mode = 'useLocalDiagNeighbor'; %useNeighbor, useLocalNeighbor, useDiagNeighbor, useLocalDiagNeighbor, useLocal, global
                               %global: take whole microstructure as feature function input, not
                               %only local window (only recommended for pooling)
linFiltSeq = false;
%load old configuration? (Optimal parameters, optimal variational distributions
loadOldConf = false;
theta_c.useNeuralNet = false;    %use neural net for p_c


%% EM params
initialIterations = 50;
basisFunctionUpdates = 10;
basisUpdateGap = 80;
maxIterations = (basisFunctionUpdates + 1)*basisUpdateGap - 1 + initialIterations;

%% Start value of model parameters
%Shape function interpolate in W
theta_cf.W = shapeInterp(domainc, domainf);
%shrink finescale domain object to save memory
domainf = domainf.shrink();
if loadOldConf
    disp('Loading old configuration...')
    theta_cf.S = dlmread('./data/S')';
    theta_cf.mu = dlmread('./data/mu')';
    theta_c.theta = dlmread('./data/theta');
    theta_c.theta = theta_c.theta(end, :)';
    s = dlmread('./data/sigma');
    s = s(end, :);
    theta_c.Sigma = sparse(diag(s));
    theta_c.SigmaInv = sparse(diag(1./s));
else
    theta_cf.S = 1e0*ones(domainf.nNodes, 1);
    theta_cf.mu = zeros(domainf.nNodes, 1);
    theta_c.theta = 0*ones(nBasis, 1);
    theta_c.Sigma = 1e-6*speye(domainc.nEl);
    s = diag(theta_c.Sigma);
    theta_c.SigmaInv = sparse(diag(1./s));
end
theta_cf.Sinv = sparse(1:domainf.nNodes, 1:domainf.nNodes, 1./theta_cf.S);
theta_cf.Sinv_vec = 1./theta_cf.S;
%precomputation to save resources
theta_cf.WTSinv = theta_cf.W'*theta_cf.Sinv;

if ~loadOldConf
    if strcmp(mode, 'useNeighbor')
        theta_c.theta = repmat(theta_c.theta, 5, 1);
    elseif strcmp(mode, 'useLocalNeighbor')
        nNeighbors = 12 + 8*(domainc.nElX - 2) + 8*(domainc.nElY - 2) +...
            5*(domainc.nElX - 2)*(domainc.nElX - 2);
        theta_c.theta = repmat(theta_c.theta, nNeighbors, 1);
    elseif strcmp(mode, 'useLocalDiagNeighbor')
        nNeighbors = 16 + 12*(domainc.nElX - 2) + 12*(domainc.nElY - 2) +...
            9*(domainc.nElX - 2)*(domainc.nElX - 2);
        theta_c.theta = repmat(theta_c.theta, nNeighbors, 1);
    elseif strcmp(mode, 'useDiagNeighbor')
        theta_c.theta = repmat(theta_c.theta, 9, 1);
    elseif strcmp(mode, 'useLocal')
        theta_c.theta = repmat(theta_c.theta, domainc.nEl, 1);
    elseif strcmp(mode, 'global')
        theta_c.theta = zeros(domainf.nEl*domainc.nEl/prod(wndw), 1); %wndw is set in genBasisFunctions
    end
end

%what kind of prior for theta_c
theta_prior_type = 'none';                  %hierarchical_gamma, hierarchical_laplace, laplace, gaussian, spikeAndSlab or none
sigma_prior_type = 'none';                  %expSigSq, delta or none. A delta prior keeps sigma at its initial value
sigma_prior_type_hold = sigma_prior_type;
fixSigInit = 0;                                 %number of initial iterations with fixed sigma
%prior hyperparams; obsolete for no prior
% theta_prior_hyperparamArray = [0 1e-4];                   %a and b params for Gamma hyperprior
theta_prior_hyperparamArray = [100];
% theta_prior_hyperparam = 10;
sigma_prior_hyperparam = 1e3*ones(domainc.nEl, 1);  %   expSigSq: x*exp(-x*sigmaSq), where x is the hyperparam

%% MCMC options
MCMC.method = 'MALA';                                %proposal type: randomWalk, nonlocal or MALA
MCMC.seed = 10;
MCMC.nThermalization = 0;                            %thermalization steps
nSamplesBeginning = [40];
MCMC.nSamples = 40;                                 %number of samples
MCMC.nGap = 40;                                     %decorrelation gap

MCMC.Xi_start = conductivityTransform(.1*condTransOpts.upperCondLim +...
    .9*condTransOpts.lowerCondLim, condTransOpts)*ones(domainc.nEl, 1);
if condTransOpts.anisotropy
    MCMC.Xi_start = ones(3*domainc.nEl, 1);
end
%only for random walk
MCMC.MALA.stepWidth = 1e-6;
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
mix_theta = 0;    %to damp oscillations/ drive convergence?



%% Variational inference params
varDistParams.mu = zeros(1, domainc.nEl);   %row vector
varDistParams.Sigma = 1e0*eye(length(varDistParams.mu));
varDistParams.sigma = ones(size(varDistParams.mu));
varDistParams.LT = chol(varDistParams.Sigma);
varDistParams.L = varDistParams.LT';
varDistParams.LInv = inv(varDistParams.L);

so{1} = StochasticOptimization('adam');
% so{1}.x = [varDistParams.mu, varDistParams.L(:)'];
% so{1}.stepWidth = [1e-2*ones(1, domainc.nEl) 1e-1*ones(1, domainc.nEl^2)];
so{1}.x = [varDistParams.mu, -2*log(varDistParams.sigma)];
so{1}.stepWidth = [1e-2*ones(1, domainc.nEl) 1*ones(1, domainc.nEl)];
so = repmat(so, nTrain, 1);

ELBOgradParams.nSamples = 10;

%Randomize among data points?
update_qi = 'sequential';    %'randomize' to randomize among data points, 'all' to update all qi's in one E-step



