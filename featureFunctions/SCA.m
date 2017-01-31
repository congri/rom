function [out] = SCA(lambda, conductivities, transform)
%Self-consistent approximation for effective conductivity, see e.g. Torquato eq. 18.14

%Binary conductivity vector
lambdaBin = (lambda > conductivities(1));
hiCondVolFrac = sum(lambdaBin)/length(lambdaBin);
loCondVolFrac = 1 - hiCondVolFrac;

alpha = conductivities(1)*(2*loCondVolFrac - 1) + conductivities(2)*(2*hiCondVolFrac - 1);

lambdaEff = .5*(alpha + sqrt(alpha^2 + 4*conductivities(1)*conductivities(2)));

if strcmp(transform, 'log')
    out = log(lambdaEff);
elseif strcmp(transform, 'logit')
    %Limitation of effective conductivity
    %Upper and lower limit on effective conductivity
    condTransOpts.upperCondLim = conductivities(2);
    condTransOpts.lowerCondLim = conductivities(1);
    condTransOpts.transform = 'logit';
    out = conductivityTransform(lambdaEff, condTransOpts);
elseif strcmp(transform, 'plain')
    out = lambdaEff;
else
    error('Which transformation for effective conductivity in SCA?')
end


end

