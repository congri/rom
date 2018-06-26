%main parameter file for 2d coarse-graining

%load old configuration? (Optimal parameters, optimal variational distributions
loadOldConf = false;

%linear filter options
rom.linFilt.type = 'local';  %local or global
rom.linFilt.gap = 0;
rom.linFilt.initialEpochs = 3;
rom.linFilt.updates = 0;     %Already added linear filters
rom.linFilt.totalUpdates = 0;
rom.maxEpochs = (rom.linFilt.totalUpdates + 1)*rom.linFilt.gap - 2 +...
    rom.linFilt.initialEpochs;


%% Start value of model parameters
%Shape function interpolate in W
rom.theta_cf.W = shapeInterp(rom.coarseMesh, rom.fineMesh);
%shrink finescale domain object to save memory
rom.fineMesh = rom.fineMesh.shrink();
if loadOldConf
    disp('Loading old configuration...')
    rom.theta_cf.S = dlmread('./data/S')';
    rom.theta_cf.mu = dlmread('./data/mu')';
    rom.theta_c.theta = dlmread('./data/theta');
    rom.theta_c.theta = rom.theta_c.theta(end, :)';
    s = dlmread('./data/sigma');
    s = s(end, :);
    rom.theta_c.Sigma = sparse(diag(s));
    rom.theta_c.SigmaInv = sparse(diag(1./s));
else
    rom.theta_cf.S = 1e3*ones(rom.fineMesh.nNodes, 1);
    rom.theta_cf.mu = zeros(rom.fineMesh.nNodes, 1);
    if rom.useAutoEnc
        load('./autoencoder/trainedAutoencoder.mat');
        latentDim = ba.latentDim;
        clear ba;
    else
        latentDim = 0;
    end
    nSecondOrderTerms = sum(sum(rom.secondOrderTerms));
    rom.theta_c.theta = 0*ones(size(rom.featureFunctions, 2) +...
        size(rom.globalFeatureFunctions,2) + latentDim + nSecondOrderTerms;
    rom.theta_c.Sigma = 1e0*speye(rom.coarseMesh.nEl);
    %s = diag(rom.theta_c.Sigma);
    %rom.theta_c.SigmaInv = sparse(diag(1./s));
    rom.theta_c.SigmaInv = inv(rom.theta_c.Sigma);
    rom.theta_c.full_Sigma = false;
end
rom.theta_cf.Sinv =...
    sparse(1:rom.fineMesh.nNodes, 1:rom.fineMesh.nNodes, 1./rom.theta_cf.S);
rom.theta_cf.Sinv_vec = 1./rom.theta_cf.S;
%precomputation to save resources
rom.theta_cf.WTSinv = rom.theta_cf.W'*rom.theta_cf.Sinv;

if ~loadOldConf
    if strcmp(rom.mode, 'useNeighbor')
        rom.theta_c.theta = repmat(rom.theta_c.theta, 5, 1);
    elseif strcmp(rom.mode, 'useLocalNeighbor')
        nNeighbors = 12 + 8*(rom.coarseMesh.nElX - 2) +...
            8*(rom.coarseMesh.nElY - 2) +...
            5*(rom.coarseMesh.nElX - 2)*(rom.coarseMesh.nElX - 2);
        rom.theta_c.theta = repmat(rom.theta_c.theta, nNeighbors, 1);
    elseif strcmp(rom.mode, 'useLocalDiagNeighbor')
        nNeighbors = 16 + 12*(rom.coarseMesh.nElX - 2) +...
            12*(rom.coarseMesh.nElY - 2) +...
            9*(rom.coarseMesh.nElX - 2)*(rom.coarseMesh.nElX - 2);
        rom.theta_c.theta = repmat(rom.theta_c.theta, nNeighbors, 1);
    elseif strcmp(rom.mode, 'useDiagNeighbor')
        rom.theta_c.theta = repmat(rom.theta_c.theta, 9, 1);
    elseif strcmp(rom.mode, 'useLocal')
        rom.theta_c.theta = repmat(rom.theta_c.theta, rom.coarseMesh.nEl, 1);
    elseif strcmp(rom.mode, 'global')
        %wndw is set in genBasisFunctions
        rom.theta_c.theta =...
            zeros(rom.fineMesh.nEl*rom.coarseMesh.nEl/prod(wndw), 1);
    end
end

%% MCMC options
%proposal type: randomWalk, nonlocal or MALA
MCMC.method = 'MALA';
MCMC.seed = 10;
MCMC.nThermalization = 0;                           %thermalization steps
nSamplesBeginning = [40];
MCMC.nSamples = 40;                                 %number of samples
MCMC.nGap = 40;                                     %decorrelation gap

MCMC.Xi_start =...
    conductivityTransform(.1*rom.conductivityTransformation.limits(2) +...
    .9*rom.conductivityTransformation.limits(1),...
    rom.conductivityTransformation)*ones(rom.coarseMesh.nEl, 1);
if rom.conductivityTransformation.anisotropy
    MCMC.Xi_start = ones(3*rom.coarseMesh.nEl, 1);
end
%only for random walk
MCMC.MALA.stepWidth = 1e-6;
stepWidth = 2e-0;
%random walk proposal covariance
MCMC.randomWalk.proposalCov = stepWidth*eye(rom.coarseMesh.nEl);
MCMC = repmat(MCMC, rom.nTrain, 1);

%% MCMC options for test chain to find step width
MCMCstepWidth = MCMC;
for i = 1:rom.nTrain
    MCMCstepWidth(i).nSamples = 2;
    MCMCstepWidth(i).nGap = 100;
end

%% Variational inference params
variationalDist = 'diagonalGauss';
if(rom.conductivityDistributionParams{1} < 0)
    %row vector
    varDistParams{1}.mu = conductivityTransform((.5*rom.upperConductivity +...
        .5*rom.lowerConductivity)*...
        ones(1, rom.coarseMesh.nEl), rom.conductivityTransformation);
else
    varDistParams{1}.mu = conductivityTransform(...
        (rom.conductivityDistributionParams{1}*rom.upperConductivity + ...
        (1 - rom.conductivityDistributionParams{1})*rom.lowerConductivity)*...
        ones(1, rom.coarseMesh.nEl), rom.conductivityTransformation);%row vector
end
if strcmp(variationalDist, 'diagonalGauss')
    varDistParams{1}.sigma = 1e2*ones(size(varDistParams{1}.mu));
elseif strcmp(variationalDist, 'fullRankGauss')
    varDistParams{1}.Sigma = 1e0*eye(length(varDistParams{1}.mu));
    varDistParams{1}.LT = chol(varDistParams{1}.Sigma);
    varDistParams{1}.L = varDistParams{1}.LT';
    varDistParams{1}.LInv = inv(varDistParams{1}.L);
end

varDistParams = repmat(varDistParams, rom.nTrain, 1);

varDistParamsVec{1} = [varDistParams{1}.mu, -2*log(varDistParams{1}.sigma)];
varDistParamsVec = repmat(varDistParamsVec, rom.nTrain, 1);

so{1} = StochasticOptimization('adam');
% so{1}.x = [varDistParams.mu, varDistParams.L(:)'];
% so{1}.stepWidth = [1e-2*ones(1, romObj.coarseMesh.nEl)...
%1e-1*ones(1, romObj.coarseMesh.nEl^2)];
so{1}.x = [varDistParams{1}.mu, -2*log(varDistParams{1}.sigma)];
sw = [1e-2*ones(1, rom.coarseMesh.nEl) 1e0*ones(1, rom.coarseMesh.nEl)];
so{1}.stepWidth = sw;
so = repmat(so, rom.nTrain, 1);

ELBOgradParams.nSamples = 10;

%Randomize among data points?
%'randomize' to rand. among data points, 'all' to update all qi's in one E-step
update_qi = 'sequential';



