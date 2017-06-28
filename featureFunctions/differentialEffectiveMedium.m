function [out] = differentialEffectiveMedium(lambda, conductivities, conductivityTransformation, matrixPhase)
%DEM approximation, Torquato eq. 18.23

lambda = lambda(:);
%Binary conductivity vector
lambdaBin = (lambda > conductivities(1));
volFrac(2) = sum(lambdaBin)/length(lambdaBin);
volFrac(1) = 1 - volFrac(2);

if strcmp(matrixPhase, 'lo')
    f = @(l) (conductivities(2) - l)*sqrt(conductivities(1)/l) - ...
        (1 - volFrac(2))*(conductivities(2) - conductivities(1));
elseif strcmp(matrixPhase, 'hi')
    f = @(l) (conductivities(1) - l)*sqrt(conductivities(2)/l) - ...
        (1 - volFrac(1))*(conductivities(1) - conductivities(2));
else
    error('DEM for high or low conducting phase as inclusion/matrix?')
end

lambdaEff = fzero(f, [conductivities(1) conductivities(2)]);


if strcmp(conductivityTransformation.type, 'log')
    out = log(lambdaEff);
elseif strcmp(conductivityTransformation.type, 'logit')
    %Limitation of effective conductivity
    out = conductivityTransform(lambdaEff, conductivityTransformation);
elseif strcmp(conductivityTransformation.type, 'plain')
    out = lambdaEff;
else
    error('Which transformation for effective conductivity in differentialEffectiveMedium?')
end


end

