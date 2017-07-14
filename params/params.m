%main parameter file for 2d coarse-graining
%CHANGE JOBFILE IF YOU CHANGE LINE NUMBERS!

%load old configuration? (Optimal parameters, optimal variational distributions
loadOldConf = false;
romObj.theta_c.useNeuralNet = false;    %use neural net for p_c


%% EM params
initialEpochs = 120;
basisFunctionUpdates = 0;
basisUpdateGap = 10;     %given in epochs, i.e. how often every data point has been seen
maxEpochs = (basisFunctionUpdates + 1)*basisUpdateGap - 2 + initialEpochs;

%% Start value of model parameters
%Shape function interpolate in W
romObj.theta_cf.W = shapeInterp(romObj.coarseScaleDomain, romObj.fineScaleDomain);
%shrink finescale domain object to save memory
romObj.fineScaleDomain = romObj.fineScaleDomain.shrink();
if loadOldConf
    disp('Loading old configuration...')
    romObj.theta_cf.S = dlmread('./data/S')';
    romObj.theta_cf.mu = dlmread('./data/mu')';
    romObj.theta_c.theta = dlmread('./data/theta');
    romObj.theta_c.theta = romObj.theta_c.theta(end, :)';
    s = dlmread('./data/sigma');
    s = s(end, :);
    romObj.theta_c.Sigma = sparse(diag(s));
    romObj.theta_c.SigmaInv = sparse(diag(1./s));
else
    romObj.theta_cf.S = 1e0*ones(romObj.fineScaleDomain.nNodes, 1);
    romObj.theta_cf.mu = zeros(romObj.fineScaleDomain.nNodes, 1);
    if romObj.useAutoEnc
        load('./autoencoder/trainedAutoencoder.mat');
        latentDim = ba.latentDim;
        clear ba;
    else
        latentDim = 0;
    end
    romObj.theta_c.theta = 0*ones(size(romObj.featureFunctions, 2) +...
        size(romObj.globalFeatureFunctions, 2) + latentDim, 1);
    romObj.theta_c.Sigma = 1e0*speye(romObj.coarseScaleDomain.nEl);
    s = diag(romObj.theta_c.Sigma);
    romObj.theta_c.SigmaInv = sparse(diag(1./s));
end
romObj.theta_cf.Sinv = sparse(1:romObj.fineScaleDomain.nNodes, 1:romObj.fineScaleDomain.nNodes, 1./romObj.theta_cf.S);
romObj.theta_cf.Sinv_vec = 1./romObj.theta_cf.S;
%precomputation to save resources
romObj.theta_cf.WTSinv = romObj.theta_cf.W'*romObj.theta_cf.Sinv;

if ~loadOldConf
    if strcmp(romObj.mode, 'useNeighbor')
        romObj.theta_c.theta = repmat(romObj.theta_c.theta, 5, 1);
    elseif strcmp(romObj.mode, 'useLocalNeighbor')
        nNeighbors = 12 + 8*(romObj.coarseScaleDomain.nElX - 2) + 8*(romObj.coarseScaleDomain.nElY - 2) +...
            5*(romObj.coarseScaleDomain.nElX - 2)*(romObj.coarseScaleDomain.nElX - 2);
        romObj.theta_c.theta = repmat(romObj.theta_c.theta, nNeighbors, 1);
    elseif strcmp(romObj.mode, 'useLocalDiagNeighbor')
        nNeighbors = 16 + 12*(romObj.coarseScaleDomain.nElX - 2) + 12*(romObj.coarseScaleDomain.nElY - 2) +...
            9*(romObj.coarseScaleDomain.nElX - 2)*(romObj.coarseScaleDomain.nElX - 2);
        romObj.theta_c.theta = repmat(romObj.theta_c.theta, nNeighbors, 1);
    elseif strcmp(romObj.mode, 'useDiagNeighbor')
        romObj.theta_c.theta = repmat(romObj.theta_c.theta, 9, 1);
    elseif strcmp(romObj.mode, 'useLocal')
        romObj.theta_c.theta = repmat(romObj.theta_c.theta, romObj.coarseScaleDomain.nEl, 1);
    elseif strcmp(romObj.mode, 'global')
        romObj.theta_c.theta = zeros(romObj.fineScaleDomain.nEl*romObj.coarseScaleDomain.nEl/prod(wndw), 1); %wndw is set in genBasisFunctions
    end
end

%what kind of prior for theta_c
romObj.theta_c.thetaPriorType = 'RVM';              %hierarchical_gamma, hierarchical_laplace, laplace,
                                                         %gaussian, RVM or none
sigma_prior_type = 'none';                  %expSigSq, delta or none. A delta prior keeps sigma at its initial value
sigma_prior_type_hold = sigma_prior_type;
fixSigInit = 0;                                 %number of initial iterations with fixed sigma
%prior hyperparams; obsolete for no prior
% theta_prior_hyperparamArray = [0 1e-4];                   %a and b params for Gamma hyperprior
romObj.theta_c.priorHyperparam = 1;
romObj.theta_c.priorHyperparam = ones(size(romObj.theta_c.theta));     %start value. this is adjusted using max marginal likelihood
% theta_prior_hyperparam = 10;
sigma_prior_hyperparam = 1e0*ones(romObj.coarseScaleDomain.nEl, 1);  %   expSigSq: x*exp(-x*sigmaSq), where x is the hyperparam

%% MCMC options
MCMC.method = 'MALA';                                %proposal type: randomWalk, nonlocal or MALA
MCMC.seed = 10;
MCMC.nThermalization = 0;                            %thermalization steps
nSamplesBeginning = [40];
MCMC.nSamples = 40;                                 %number of samples
MCMC.nGap = 40;                                     %decorrelation gap

MCMC.Xi_start = conductivityTransform(.1*romObj.conductivityTransformation.limits(2) +...
    .9*romObj.conductivityTransformation.limits(1), romObj.conductivityTransformation)*ones(romObj.coarseScaleDomain.nEl, 1);
if romObj.conductivityTransformation.anisotropy
    MCMC.Xi_start = ones(3*romObj.coarseScaleDomain.nEl, 1);
end
%only for random walk
MCMC.MALA.stepWidth = 1e-6;
stepWidth = 2e-0;
MCMC.randomWalk.proposalCov = stepWidth*eye(romObj.coarseScaleDomain.nEl);   %random walk proposal covariance
MCMC = repmat(MCMC, romObj.nTrain, 1);

%% MCMC options for test chain to find step width
MCMCstepWidth = MCMC;
for i = 1:romObj.nTrain
    MCMCstepWidth(i).nSamples = 2;
    MCMCstepWidth(i).nGap = 100;
end

%% Control convergence velocity - take weighted mean of adjacent parameter estimates
mix_sigma = 0;
mix_S = 0;
mix_W = 0;
mix_theta = 0;    %to damp oscillations/ drive convergence?



%% Variational inference params
variationalDist = 'diagonalGauss';
if(romObj.conductivityDistributionParams{1} < 0)
    varDistParams{1}.mu = conductivityTransform((.5*romObj.upperConductivity + .5*romObj.lowerConductivity)*...
    ones(1, romObj.coarseScaleDomain.nEl), romObj.conductivityTransformation);   %row vector
else
    varDistParams{1}.mu = conductivityTransform((romObj.conductivityDistributionParams{1}*romObj.upperConductivity + ...
        (1 - romObj.conductivityDistributionParams{1})*romObj.lowerConductivity)*...
        ones(1, romObj.coarseScaleDomain.nEl), romObj.conductivityTransformation);   %row vector
end
if strcmp(variationalDist, 'diagonalGauss')
    varDistParams{1}.sigma = ones(size(varDistParams{1}.mu));
elseif strcmp(variationalDist, 'fullRankGauss')
    varDistParams{1}.Sigma = 1e0*eye(length(varDistParams{1}.mu));
    varDistParams{1}.LT = chol(varDistParams{1}.Sigma);
    varDistParams{1}.L = varDistParams{1}.LT';
    varDistParams{1}.LInv = inv(varDistParams{1}.L);
end
varDistParams = repmat(varDistParams, romObj.nTrain, 1);

varDistParamsVec{1} = [varDistParams{1}.mu, -2*log(varDistParams{1}.sigma)];
varDistParamsVec = repmat(varDistParamsVec, romObj.nTrain, 1);

so{1} = StochasticOptimization('adam');
% so{1}.x = [varDistParams.mu, varDistParams.L(:)'];
% so{1}.stepWidth = [1e-2*ones(1, romObj.coarseScaleDomain.nEl) 1e-1*ones(1, romObj.coarseScaleDomain.nEl^2)];
so{1}.x = [varDistParams{1}.mu, -2*log(varDistParams{1}.sigma)];
sw = [1.2e-2*ones(1, romObj.coarseScaleDomain.nEl) 3*ones(1, romObj.coarseScaleDomain.nEl)];
so{1}.stepWidth = sw;
so = repmat(so, romObj.nTrain, 1);

ELBOgradParams.nSamples = 10;

%Randomize among data points?
update_qi = 'sequential';    %'randomize' to randomize among data points, 'all' to update all qi's in one E-step



