function [out] = maxwellGarnett(lambda, conductivities, conductivityTransformation, matrixPhase)
%Computes the effective conductivity using the Maxwell-Garnett Formula (see e.g. Torquato eq. 18.5)
%   lambda:     vector of conductivities
%   transform:  transformation of conductivity, can be 'log', 'logit' (sigmoid) or 'plain'

%ONLY FOR BINARY MATERIALS!
lambda = lambda(:);

%Binary conductivity vector
lambdaBin = (lambda > conductivities(1));
hiCondVolFrac = sum(lambdaBin)/length(lambdaBin);

if strcmp(matrixPhase, 'hi')
    %simplify notation
    lambdaMatrix = conductivities(2);
    lambdaInclusions = conductivities(1);
    inclusionVolFrac = 1 - hiCondVolFrac;
    
    lambdaEff = lambdaMatrix*((lambdaMatrix + lambdaInclusions + inclusionVolFrac*(lambdaInclusions - lambdaMatrix))/...
        (lambdaMatrix + lambdaInclusions - inclusionVolFrac*(lambdaInclusions - lambdaMatrix)));
elseif strcmp(matrixPhase, 'lo')
    %simplify notation
    lambdaMatrix = conductivities(1);
    lambdaInclusions = conductivities(2);
    inclusionVolFrac = hiCondVolFrac;
    
    lambdaEff = lambdaMatrix*((lambdaMatrix + lambdaInclusions + inclusionVolFrac*(lambdaInclusions - lambdaMatrix))/...
        (lambdaMatrix + lambdaInclusions - inclusionVolFrac*(lambdaInclusions - lambdaMatrix)));
else
    error('What is matrix phase for Maxwell-Garnett?')
end

if strcmp(conductivityTransformation.type, 'log')
    out = log(lambdaEff);
elseif strcmp(conductivityTransformation.type, 'logit')
    %Limitation of effective conductivity
    out = conductivityTransform(lambdaEff, conductivityTransformation);
elseif strcmp(conductivityTransformation.type, 'plain')
    out = lambdaEff;
else
    error('Which transformation for effective conductivity in Maxwell-Garnett?')
end

end

