function [out] = MCMCsampler(log_distribution, startValue, opts)
%Standalone MCMC sampler. Pass distribution, starting value and options and
%get samples
%By C. Grigo, July 2016
% log_distribution:             function handle to log probability distribution and its
%                               gradient; gradient as column vector
% startValue:                   Initial value of Markov chain; col. vec.
% opts:                         MCMC sampling options structure
%
% opts:
% opts.method                   proposal type: randomWalk, nonlocal or MALA
% opts.nThermalization          thermalization steps
% opts.nSamples                 number of samples
% 
% %only for random walk
% opts.randomWalk.proposalCov   random walk proposal covariance
% 
% %only for nonlocal
% opts.nonlocal.rnd             random number generator for nonlocal proposals
% opts.nonlocal.pdf             corresponding proposal pdf
% opts.nonlocal.propMean        proposal pdf is Gaussian: mean is propMean
% opts.nonlocal.propCov         proposal pdf is Gaussian: cov is propCov
%
% only for MALA
% opts.MALA.stepWidth           step size parameter


rng(opts.seed);     %random number seed based on system time


%preallocation of samples array
out.samples = zeros(size(startValue, 1), opts.nSamples);
out.log_p = zeros(opts.nSamples, 1);
out.data = cell(opts.nSamples, 1);
samplesTherm = zeros(size(startValue, 1), opts.nThermalization);
samplesTherm(:, 1) = startValue;

x = startValue;

accepted = 0;

if(strcmp(opts.method, 'MALA'))
    
   zeroMean = zeros(1, size(x, 1));
   unitCov = eye(size(x, 1));
   invProposalCov = (opts.MALA.stepWidth^(-2))*unitCov;
   %data allows to pass further data
   [log_p, d_log_p, data] = log_distribution(x);

else
    
    [log_p, data] = log_distribution(x);

end


%Thermalization
for i = 1:(opts.nThermalization - 1)

    if(strcmp(opts.method, 'randomWalk'))
        %Gaussian random walk MCMC
        
        xProp = mvnrnd(x, opts.randomWalk.proposalCov)';
        log_pProp = log_distribution(xProp);
        Metropolis = exp(log_pProp - log_p);
        
    elseif(strcmp(opts.method, 'nonlocal'))
        %"Nonlocal" proposal distribution
        
        xProp = opts.nonlocal.rnd(opts.nonlocal.propMean, opts.nonlocal.propCov);
        log_pProp = log_distribution(xProp);
        Metropolis = exp(log_pProp - log_p)*...
            ((opts.nonlocal.pdf(x, opts.nonlocal.propMean, opts.nonlocal.propCov))/...
            (opts.nonlocal.pdf(xProp, opts.nonlocal.propMean, opts.nonlocal.propCov)));
        
    elseif strcmp(opts.method, 'MALA')
        %Metropolis adjusted Langevin algorithm
        
        proposalMean = x + .5*opts.MALA.stepWidth^2*d_log_p;
        xProp = proposalMean + opts.MALA.stepWidth*mvnrnd(zeroMean, unitCov)';
        proposalExponent = -.5*(xProp - proposalMean)'*invProposalCov*(xProp - proposalMean);
        
        [log_pProp, d_log_pProp] = log_distribution(xProp);
        inverseProposalMean = xProp  + .5*opts.MALA.stepWidth^2*d_log_pProp;
        invProposalExponent = -.5*(x - inverseProposalMean)'*invProposalCov*(x - inverseProposalMean);
        
        Metropolis = exp(invProposalExponent - proposalExponent + log_pProp - log_p);
     
    else
        
        error('unknown MCMC sampling method')
        
    end

    r = rand;
    if(r < Metropolis)
        %Metropolis acceptance. Go to xProp

        x = xProp;
        log_p = log_pProp;
        if(strcmp(opts.method, 'MALA'))
            d_log_p = d_log_pProp;
        end
        accepted = accepted + 1;

    end
    samplesTherm(:, i + 1) = x;

end

acceptance = accepted/opts.nThermalization;
if(0)
    %refine proposal params after thermalization
    if(strcmp(opts.method, 'randomWalk'))
        
        if(acceptance)
            opts.randomWalk.proposalCov = (1/.7)*acceptance*opts.randomWalk.proposalCov;
        else
            opts.randomWalk.proposalCov = .2*opts.randomWalk.proposalCov;
        end
        
    elseif(strcmp(opts.method, 'nonlocal'))
        
        opts.nonlocal.propMean = mean(samplesTherm);
        %ATTENTION: fully decorrelated thermalization required!
        opts.nonlocal.propCov = .2*cov(samplesTherm) + .8*opts.nonlocal.propCov;
        
    elseif(strcmp(opts.method, 'MALA'))
        %Metropolis adjusted Langevin algorithm
        
        if(acceptance)
            opts.MALA.stepWidth = (1/.7)*acceptance*opts.MALA.stepWidth;
        else
            opts.MALA.stepWidth = .2*opts.MALA.stepWidth;
        end
        
    else
        error('unknown MCMC sampling method')
    end
end



%Actual sampling
accepted = 0;
j = 1;
for i = 1:(opts.nSamples*(opts.nGap + 1))

    if(strcmp(opts.method, 'randomWalk'))
        %Gaussian random walk MCMC

        xProp = mvnrnd(x, opts.randomWalk.proposalCov)';
        [log_pProp, dataProp] = log_distribution(xProp);
        Metropolis = exp(log_pProp - log_p);

    elseif(strcmp(opts.method, 'nonlocal'))
        %"Nonlocal" proposal distribution
        
        xProp = opts.nonlocal.rnd(opts.nonlocal.propMean, opts.nonlocal.propCov);
        [log_pProp, dataProp] = log_distribution(xProp);
        Metropolis = exp(log_pProp - log_p)*...
            ((opts.nonlocal.pdf(x, opts.nonlocal.propMean, opts.nonlocal.propCov))/...
            (opts.nonlocal.pdf(xProp, opts.nonlocal.propMean, opts.nonlocal.propCov)));
        
        
    elseif(strcmp(opts.method, 'MALA'))
        %Metropolis adjusted Langevin algorithm
        
        proposalMean = x + .5*opts.MALA.stepWidth^2*d_log_p;
        xProp = proposalMean + opts.MALA.stepWidth*mvnrnd(zeroMean, unitCov)';
        proposalExponent = -.5*(xProp - proposalMean)'*invProposalCov*(xProp - proposalMean);
        
        [log_pProp, d_log_pProp, dataProp] = log_distribution(xProp);

        inverseProposalMean = xProp  + .5*opts.MALA.stepWidth^2*d_log_pProp;
        invProposalExponent = -.5*(x - inverseProposalMean)'*invProposalCov*(x - inverseProposalMean);

        Metropolis = exp(invProposalExponent - proposalExponent + log_pProp - log_p);
        
    else
        
        error('unknown MCMC sampling method')
        
    end

    r = rand;
    if r < Metropolis
        %Metropolis acceptance. Go to xProp

        x = xProp;
        log_p = log_pProp;
        data = dataProp;
        if(strcmp(opts.method, 'MALA'))
            d_log_p = d_log_pProp;
        end
        accepted = accepted + 1;
%         currAcc = accepted/i

    end
    if(~mod(i, (opts.nGap + 1)))
        out.samples(:, j) = x;
        out.log_p(j) = log_p;
        out.data{j} = data;
        j = j + 1;
    end

end

out.acceptance = accepted/(opts.nSamples*(opts.nGap + 1));
if out.acceptance < .2
    warning('Acceptance ratio is')
    acc = out.acceptance
    log_p
    log_pProp
%     pause
end
out.log_pEnd = log_p;




end

