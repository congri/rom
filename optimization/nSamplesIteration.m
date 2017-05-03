function [nSamples] = nSamplesIteration(iteration, minSamples, maxSamples)
%Function that gives the number of gradient samples in stochastic gradient descent
%'iteration' starts at 1

mode = 'poly';
if strcmp(mode, 'poly')
    exponent = 1;
    coeff = 20;
    nSamples = round(minSamples + coeff*(iteration - 1)^exponent);
else
    error('unknown mode')
end

if nSamples < minSamples
    nSamples = minSamples;
end

if nSamples > maxSamples
    nSamples = maxSamples;
end

end

