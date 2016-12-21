%Script to collect data in data arrays of EM object

if ~exist('./data/', 'dir')
    mkdir('./data/');
end
%Remove old data in first step, if there exists some
if(k == 1)
    delete('./data/MCMCstepWidth', './data/sigma', './data/S', './data/mu', './data/theta', './data/Wmat')
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
    [rowW, colW, valW] = find(theta_cf.W);
    WArray = [rowW, colW, valW]';
    onlyFinal = true;
    if onlyFinal
        save(filename, 'WArray', '-ascii')
    else
        save(filename, 'WArray', '-ascii', '-append')
    end
    clear rowW colW valW WArray;
end

%theta
filename = './data/theta';
theta = theta_c.theta';
save(filename, 'theta', '-ascii', '-append');

%sigma
filename = './data/sigma';
sigma = theta_c.sigma;
save(filename, 'sigma', '-ascii', '-append');

%S
saveS = true;
if saveS
    filename = './data/S';
    S = theta_cf.S';
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
    mu = theta_cf.mu';
    filename = './data/mu';
    onlyFinal = true;
    if onlyFinal
        save(filename, 'mu', '-ascii')
    else
        save(filename, 'mu', '-ascii', '-append')
    end
    clear mu;
end
