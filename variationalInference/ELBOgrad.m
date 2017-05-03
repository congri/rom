function [grad, gradErr] = ELBOgrad(variationalParams, trueLogDist, params, nSamples)
%Monte Carlo estimate of ELBO gradient
%   samples:                samples from standard normal

if strcmp(params.family, 'diagonalGaussian')
    dim = length(variationalParams)/2;
    variationalMean = variationalParams(1:dim);
    %variationalParams holds optimal log sigma^-2
    variationalStd = exp(-.5*variationalParams(((dim + 1):end)));
    meanGrad = zeros(1, dim);
    varGrad = zeros(1, dim);
    
    meanGradSq = zeros(1, dim);
    varGradSq = zeros(1, dim);
    for i = 1:nSamples
        sample = normrnd(0, 1, 1, dim);
%         variationalSample = variationalMean + variationalStd.*samples(i, :);
        variationalSample = variationalMean + variationalStd.*sample;
        [~, trueGrad] = trueLogDist(variationalSample);
        trueGrad = trueGrad';
        %Gradient of model distribution
        meanGrad = ((i - 1)/i)*meanGrad + (1/i)*trueGrad;
%         varGrad = ((i - 1)/i)*varGrad + (1/i)*trueGrad.*samples(i, :);
        %gradient w.r.t. d/d_sigma_k
        varGrad = ((i - 1)/i)*varGrad + (1/i)*(trueGrad.*sample + 1./variationalStd);

        if nargout > 1
            %Might be untrue as gradient of variational distribution is missing
            meanGradSq = ((i - 1)/i)*meanGradSq + (1/i)*(trueGrad).^2;
%             varGradSq = ((i - 1)/i)*varGradSq + (1/i)*((trueGrad).*samples(i, :)).^2;
            %w.r.t. d/d_sigma_k
            varGradSq = ((i - 1)/i)*varGradSq + (1/i)*(trueGrad.*sample + 1./variationalStd).^2;
        end
    end
    
    %Modification due to gradient of variational distribution (given analytically)
    %     varGrad = varGrad + 1./variationalStd;
    
    %Due to log transformation
    varGrad = -.5*(varGrad.*variationalStd);
    grad = [meanGrad varGrad];
    
    if nargout > 1
        MeanGradErr = sqrt(abs(meanGradSq - meanGrad.^2))/sqrt(nSamples);
        %error w.r.t. d/d_sigma_k
        VarGradErr = sqrt(.25*(variationalStd.^2).*varGradSq - varGrad.^2)/sqrt(nSamples);
        gradErr = [MeanGradErr VarGradErr];
    end
end


end

