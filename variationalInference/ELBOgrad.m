function [grad, gradErr] = ELBOgrad(samples, variationalParams, trueLogDist, params)
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
    for i = 1:params.nSamples
        variationalSample = variationalMean + variationalStd.*samples(i, :);
        try
            [~, trueGrad] = trueLogDist(variationalSample);
        catch
            error('Gradient only implemented with reparametrization trick')
        end
        trueGrad = trueGrad';
        %Gradient of model distribution
        meanGrad = ((i - 1)/i)*meanGrad + (1/i)*trueGrad;
        varGrad = ((i - 1)/i)*varGrad + (1/i)*trueGrad.*samples(i, :);

        if nargout > 1
            %Might be untrue as gradient of variational distribution is missing
            meanGradSq = ((i - 1)/i)*meanGradSq + (1/i)*(trueGrad).^2;
            varGradSq = ((i - 1)/i)*varGradSq + (1/i)*((trueGrad).*samples(i, :)).^2;
        end
    end
    
    %Modification due to gradient of variational distribution (given analytically)
    varGrad = varGrad + variationalStd;
    
    %Due to log transformation
    varGrad = -.5*(varGrad.*variationalStd);
    grad = [meanGrad varGrad];
    if nargout > 1
        MeanGradErr = real(sqrt(meanGradSq - meanGrad.^2))/params.nSamples;
        VarGradErr = real(sqrt(varGradSq - varGrad.^2))/params.nSamples;
        gradErr = [MeanGradErr VarGradErr];
    end
end


end

