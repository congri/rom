function [sampleFun] = genBochnerSamples(lengthScale, sigma_f2, nBasisFunctions)
%Generate approximate Gaussian process sample functions in analytical form using Bochner's theorem

%Stacked samples from W, see reference_notes
W = mvnrnd(zeros(1, 2), lengthScale^(-2)*eye(2), nBasisFunctions);

%Stacked samples from b, see notes
b = 2*pi*rand(nBasisFunctions, 1);

%Draw coefficients gamma
gamma = normrnd(0, 1, 1, nBasisFunctions);

%Handle to sample function
sampleFun = @(x) sqrt((2*sigma_f2)/nBasisFunctions)*(gamma*cos(W*x + b));


end

