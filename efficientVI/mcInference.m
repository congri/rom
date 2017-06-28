function E = mcInference(functionHandle, variationalDist, varDistParams)
%Samples expected value of function in function handle under variational dist.
inferenceSamples = 100;
if strcmp(variationalDist, 'diagonalGauss')
    %individual samples are rows of 'samples'
    %sigma is std
    samples = mvnrnd(varDistParams.mu, varDistParams.sigma, inferenceSamples);
    E = 0;
    for i = 1:inferenceSamples
        E = (1/i)*((i - 1)*E + functionHandle(samples(i, :)));
    end
elseif strcmp(variationalDist, 'fullRankGauss')
    %individual samples are rows of 'samples'
    %Sigma is covariance
    samples = mvnrnd(varDistParams.mu, varDistParams.Sigma, inferenceSamples);
    E = 0;
    for i = 1:inferenceSamples
        E = (1/i)*((i - 1)*E + functionHandle(samples(i, :)));
    end
else
    error('Unknown variational distribution')
end
end

