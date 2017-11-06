%Script to collect data in data arrays of EM object

if ~exist('./data/', 'dir')
    mkdir('./data/');
end
%Remove old data in first step, if there exists some
if(romObj.EM_iterations == 1)
    delete('./data/MCMCstepWidth', './data/sigma', './data/S', './data/mu',...
        './data/theta', './data/Wmat', './data/w', './data/E', './data/neighborDictionary', './noPriorSigma.mat')
%     rmdir('./data', 's')
%     mkdir('./data')
end

%% MCMC Step width
saveSW = false;
if saveSW
    MCMCStepWidth = zeros(1, nTrain);
    filename = './data/MCMCstepWidth';
    for i = 1:nTrain
        if strcmp(MCMC(i).method, 'MALA')
            MCMCStepWidth(i) = MCMC(i).MALA.stepWidth;
        elseif strcmp(MCMC(i).method, 'randomWalk')
            %only valid for isotropic proposals!
            MCMCStepWidth(i) = MCMC(i).randomWalk.proposalCov(1, 1);
        elseif strcmp(MCMC(i).method, 'nonlocal')
            %do nothing; we don't use this
        else
            error('Unknown sampling method')
        end
    end
    save(filename, 'MCMCStepWidth', '-ascii', '-append')
end

%% Optimal params
%W matrix
saveW = true;
if saveW
    filename = './data/Wmat';
    [rowW, colW, valW] = find(romObj.theta_cf.W);
    WArray = [rowW, colW, valW]';
    onlyFinal = true;
    if onlyFinal
        save(filename, 'WArray', '-ascii')
    else
        save(filename, 'WArray', '-ascii', '-append')
    end
    clear rowW colW valW WArray;
end

%lambda
filename = './data/thetaPriorHyperparam';
thetaPriorHyperparam = romObj.thetaPriorHyperparam';
save(filename, 'thetaPriorHyperparam', '-ascii', '-append');

%theta
filename = './data/theta';
theta = romObj.theta_c.theta';
save(filename, 'theta', '-ascii', '-append');

%sigma
filename = './data/sigma';
if romObj.theta_c.full_Sigma
    sigma = romObj.theta_c.Sigma(:)';
else
    sigma = full(diag(romObj.theta_c.Sigma))';
end
save(filename, 'sigma', '-ascii', '-append');

%S
saveS = true;
if saveS
    filename = './data/S';
    S = romObj.theta_cf.S';
    onlyFinal = true;
    if onlyFinal
        save(filename, 'S', '-ascii');
    else
        save(filename, 'S', '-ascii', '-append');
    end
    clear S;
end
%mu
saveMu = true;
if saveMu
    mu = romObj.theta_cf.mu';
    filename = './data/mu';
    onlyFinal = true;
    if onlyFinal
        save(filename, 'mu', '-ascii')
    else
        save(filename, 'mu', '-ascii', '-append')
    end
    clear mu;
end

%neighbor dictionary
saveNeighborDictionary = true;
if saveNeighborDictionary
    nbdict = romObj.neighborDictionary;
    filename = './data/neighborDictionary';
    save(filename, 'nbdict', '-ascii');
    clear nbdict;
end
