function [out] = SCA(lambda, conductivities, conductivityTransformation)
%Self-consistent approximation for effective conductivity, see e.g. Torquato eq. 18.14

lambda = lambda(:);

%Binary conductivity vector
lambdaBin = (lambda > conductivities(1));
hiCondVolFrac = sum(lambdaBin)/length(lambdaBin);
loCondVolFrac = 1 - hiCondVolFrac;

alpha = conductivities(1)*(2*loCondVolFrac - 1) + conductivities(2)*(2*hiCondVolFrac - 1);

lambdaEff = .5*(alpha + sqrt(alpha^2 + 4*conductivities(1)*conductivities(2)));

if (strcmp(conductivityTransformation.type, 'log') || strcmp(conductivityTransformation.type, 'logit') ||...
    strcmp(conductivityTransformation.type, 'square'))
    out = conductivityTransform(lambdaEff, conductivityTransformation);
elseif strcmp(conductivityTransformation.type, 'plain')
    out = lambdaEff;
else
    error('Which transformation for effective conductivity in SCA?')
end


end

