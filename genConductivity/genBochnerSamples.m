function [sampleFun] = genBochnerSamples(lengthScale, sigma_f2, nBasisFunctions, type)
%Generate approximate Gaussian process sample functions in analytical form using Bochner's theorem

if strcmp(type, 'squaredExponential')
    %Stacked samples from W, see reference_notes
    W = mvnrnd(zeros(1, 2), diag(lengthScale.^(-2)), nBasisFunctions);
elseif strcmp(type, 'ornsteinUhlenbeck')
    W = trnd(1, nBasisFunctions, 1)/lengthScale(1);
    W = [W, trnd(1, nBasisFunctions, 1)/lengthScale(2)];
else
    error('Unknown covariance type')
end

%Stacked samples from b, see notes
b = 2*pi*rand(nBasisFunctions, 1);

%Draw coefficients gamma
gamma = normrnd(0, 1, 1, nBasisFunctions);

%Handle to sample function
sampleFun = @(x) sqrt((2*sigma_f2)/nBasisFunctions)*(gamma*cos(W*x + b));


end

