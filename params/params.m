%main parameter file for 2d coarse-graining

%load old configuration? (Optimal parameters, optimal variational distributions
loadOldConf = false;

%linear filter options
romObj.linFilt.type = 'local';  %local or global
romObj.linFilt.gap = 0;
romObj.linFilt.initialEpochs = 100;
romObj.linFilt.updates = 0;     %Already added linear filters
romObj.linFilt.totalUpdates = 0;
romObj.maxEpochs = (romObj.linFilt.totalUpdates + 1)*romObj.linFilt.gap - 2 + romObj.linFilt.initialEpochs;


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
    romObj.theta_cf.S = 1e3*ones(romObj.fineScaleDomain.nNodes, 1);
    romObj.theta_cf.mu = zeros(romObj.fineScaleDomain.nNodes, 1);
    if romObj.useAutoEnc
        load('./autoencoder/trainedAutoencoder.mat');
        latentDim = ba.latentDim;
        clear ba;
    else
        latentDim = 0;
    end
    nSecondOrderTerms = sum(sum(romObj.secondOrderTerms));
    romObj.theta_c.theta = 0*ones(size(romObj.featureFunctions, 2) +...
        size(romObj.globalFeatureFunctions, 2) + latentDim + nSecondOrderTerms + ...
        size(romObj.convectionFeatureFunctions, 2) + size(romObj.globalConvectionFeatureFunctions, 2), 1);
    if romObj.useConvection
        romObj.theta_c.Sigma = 1e0*speye(3*romObj.coarseScaleDomain.nEl);
    else
        romObj.theta_c.Sigma = 1e0*speye(romObj.coarseScaleDomain.nEl);
    end
%     s = diag(romObj.theta_c.Sigma);
%     romObj.theta_c.SigmaInv = sparse(diag(1./s));
    romObj.theta_c.SigmaInv = inv(romObj.theta_c.Sigma);
    romObj.theta_c.full_Sigma = false;
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
    varDistParams{1}.sigma = 1e2*ones(size(varDistParams{1}.mu));
    if romObj.useConvection
        varDistParams{1}.sigma = ones(1, 3*romObj.coarseScaleDomain.nEl);
        %Sharp convection field at 0 at beginning
        varDistParams{1}.sigma((romObj.coarseScaleDomain.nEl + 1):end) = 1e-2;
    end
elseif strcmp(variationalDist, 'fullRankGauss')
    varDistParams{1}.Sigma = 1e0*eye(length(varDistParams{1}.mu));
    varDistParams{1}.LT = chol(varDistParams{1}.Sigma);
    varDistParams{1}.L = varDistParams{1}.LT';
    varDistParams{1}.LInv = inv(varDistParams{1}.L);
end

if romObj.useConvection
    varDistParams{1}.mu = [varDistParams{1}.mu, zeros(1, 2*romObj.coarseScaleDomain.nEl)];
end

varDistParams = repmat(varDistParams, romObj.nTrain, 1);

varDistParamsVec{1} = [varDistParams{1}.mu, -2*log(varDistParams{1}.sigma)];
varDistParamsVec = repmat(varDistParamsVec, romObj.nTrain, 1);

so{1} = StochasticOptimization('adam');
% so{1}.x = [varDistParams.mu, varDistParams.L(:)'];
% so{1}.stepWidth = [1e-2*ones(1, romObj.coarseScaleDomain.nEl) 1e-1*ones(1, romObj.coarseScaleDomain.nEl^2)];
so{1}.x = [varDistParams{1}.mu, -2*log(varDistParams{1}.sigma)];
sw = [1e-2*ones(1, romObj.coarseScaleDomain.nEl) 1e0*ones(1, romObj.coarseScaleDomain.nEl)];
if romObj.useConvection
    sw = [1e-2*ones(1, romObj.coarseScaleDomain.nEl) ...    %conductivity mean
        1e-4*ones(1, 2*romObj.coarseScaleDomain.nEl) ...    %advection mean
        1e-0*ones(1, romObj.coarseScaleDomain.nEl) ...      %conductivity sigma
        1e-2*ones(1, 2*romObj.coarseScaleDomain.nEl)];      %advection sigma
end
so{1}.stepWidth = sw;
so = repmat(so, romObj.nTrain, 1);

ELBOgradParams.nSamples = 10;

%Randomize among data points?
update_qi = 'sequential';    %'randomize' to randomize among data points, 'all' to update all qi's in one E-step



