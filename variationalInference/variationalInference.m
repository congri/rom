function [optVarDist, steps] = variationalInference(logTrueCondDist, params, initialParams)
%Function for variational inference
%Input:
%   trueDist:           handle to true conditional distribution p(hidden| observable)
%   family:             family of variational distributions
%   initialParams:      parameter initialization for variational family





if strcmp(params.family, 'isotropicGaussian')
    %BUGGED!!!
    variationalGaussMean = initialParams{1};
    dim = length(initialParams{1});
    varVariance = initialParams{2};
    mlog_varVariance = -log(varVariance);
    GMean = zeros(1, dim);
    GVar = 0;
    
    converged = false;
    steps = 0;
    RMsamples_mu = zeros(params.nSamples, dim);
    RMsamples_var = zeros(params.nSamples, 1);
    while ~converged
        varMean = repmat(variationalGaussMean, params.nSamples, 1);
        %Draw samples from variational distribution
        samples = normrnd(varMean, varVariance); %1 sample = 1 row
        
        %Gradient of variational distribution w.r.t. params
        samplesMinusMu = (samples - varMean);
        samplesMinusMuNormSq = sum(samplesMinusMu.^2, 2);
        d_muSamples = (1/varVariance)*samplesMinusMu;
        d_sigmaMinus2Samples = .5*dim*varVariance - .5*samplesMinusMuNormSq;
        
        for i = 1:params.nSamples
            rightFactor = logTrueCondDist(samples(i, :)) + .5*dim*log(2*pi) +...
                .5*dim*log(varVariance) + .5*(1/varVariance)*samplesMinusMuNormSq(i, :);
            RMsamples_mu(i, :) = d_muSamples(i, :)*rightFactor;
            %factor 1/varVariance due to log transformation
            RMsamples_var(i) = (1/varVariance)*d_sigmaMinus2Samples(i)*rightFactor;
        end
        
        %Robbins-Monro
        variationalGaussMean = mean(varMean);
        oldParams = [variationalGaussMean varVariance];
        meanGrad = mean(RMsamples_mu);
        GMean = GMean + meanGrad.^2;    %AdaGrad
        rhoMean = params.robbinsMonro.stepWidth*GMean.^(-.5);
        %         varMeanMean = varMeanMean + (params.robbinsMonro.stepWidth/(params.robbinsMonro.offset + RMsteps))*...
        %             mean(RMsamples_mu);
        variationalGaussMean = variationalGaussMean + rhoMean.*meanGrad;
        varGrad = mean(RMsamples_var);
        GVar = GVar + varGrad^2;    %AdaGrad
        rhoVar = params.robbinsMonro.stepWidth*GVar^(-.5);
        mlog_varVariance = mlog_varVariance + rhoVar*varGrad;
        varVariance = 1/(exp(mlog_varVariance));
        %converged?
        if (norm(oldParams - [variationalGaussMean varVariance])/norm([variationalGaussMean varVariance]) < params.robbinsMonro.relXtol)
            converged = true;
        end
        steps = steps + 1;
    end
    
    %Construct function handle to optimal variational distribution
    optVarDist.dist = @(x) normpdf(x, variationalGaussMean, sqrt(varVariance));
    optVarDist.params{1} = variationalGaussMean;
    optVarDist.params{2} = varVariance;
    
elseif strcmp(params.family, 'diagonalGaussian')
    
    variationalParameters = initialParams;
    
    gradFunc = @(samp, varParam) ELBOgrad(samp, varParam, logTrueCondDist, params);
    sampleFunc = @(varParam) normrnd(0, 1, params.nSamples, length(varParam)/2);
    [variationalParameters, steps] = stochasticOptimization(variationalParameters, gradFunc, sampleFunc, params.optParams)
    
    %Construct function handle to optimal variational distribution
    variationalGaussSigma = exp(-.5*variationalParameters((params.optParams.dim + 1):end));
    optVarDist.dist = @(x) normpdf(x, variationalParameters(1:params.optParams.dim), sqrt(variationalGaussSigma));
    %variationalParameters = [mean, log(sigma^-2)]
    optVarDist.params = variationalParameters;
    
elseif strcmp(params.family, 'fullRankGaussian')
    %BUGGED!!!
    varMean = initialParams{1};
    dim = length(initialParams{1});
    varCovar = initialParams{2};
    varPrecision = inv(varCovar);
    L = chol(varPrecision);
    GMean = zeros(1, dim);
    GL = zeros(dim);
    logpi = .5*dim*log(2*pi);
    
    converged = false;
    steps = 0;
    RMsamples_mu = zeros(params.nSamples, dim);
    RMsamples_L = zeros(dim, dim, params.nSamples);
    samplesMinusMu = zeros(params.nSamples, dim);
    d_LSamples = zeros(dim, dim, params.nSamples);
    while ~converged
        %Draw samples from variational distribution
        samples = mvnrnd(varMean, varCovar, params.nSamples); %1 sample = 1 row
        
        %Gradient of variational distribution w.r.t. params
        for i = 1:params.nSamples
            samplesMinusMu(i, :) = (samples(i, :) - varMean);
        end
        d_muSamples = samplesMinusMu*varPrecision;
        %L is an upper triangle Cholesky factorization matrix of varPrecision
        invLT = inv(L)';
        for i = 1:params.nSamples
            d_LSamples(:, :, i) = invLT - (samplesMinusMu(i, :)'*samplesMinusMu(i, :))*L;
        end
        
        logdetL = sum(log(diag(abs(L))));
        for i = 1:params.nSamples
            %The right factor in eq. 3 in Ranganath, Gerrish, Blei
            rightFactor = logTrueCondDist(samples(i, :)) + logpi - logdetL...
                + .5*samplesMinusMu(i, :)*varPrecision*samplesMinusMu(i, :)';
            RMsamples_mu(i, :) = d_muSamples(i, :)*rightFactor;
            RMsamples_L(:, :, i) = d_LSamples(:, :, i)*rightFactor;
        end
        
        %Robbins-Monro
        oldParams = [varMean(:); varCovar(:)];
        meanGrad = mean(RMsamples_mu);
        GMean = GMean + meanGrad.^2;    %AdaGrad
        rhoMean = params.robbinsMonro.stepWidth*GMean.^(-.5);
        varMean = varMean + rhoMean.*meanGrad;
        LGrad = mean(RMsamples_L, 3);
        GL = GL + LGrad.^2;
        rhoL = params.robbinsMonro.stepWidth*GL.^(-.5);
        %Add stability term to avoid singular matrix L
        L = L + rhoL.*LGrad + (1e-10)*eye(dim);
        varPrecision = L'*L;
        varCovar = inv(varPrecision);
        
        %converged?
        if (norm(oldParams - [varMean(:); varCovar(:)])/norm([varMean(:); varCovar(:)]) < params.robbinsMonro.relXtol)
            converged = true;
        end
        steps = steps + 1;
    end
    
    %Construct function handle to optimal variational distribution
    optVarDist.dist = @(x) mvnpdf(x, varMean, varCovar);
    optVarDist.params{1} = varMean;
    optVarDist.params{2} = varCovar;
    
else
    error('Unknown parametric family for variational inference')
end

end

