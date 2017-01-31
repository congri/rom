function [out] = differentialEffectiveMedium(lambda, conductivities, transform, hilo)
%DEM approximation, Torquato eq. 18.23

%Binary conductivity vector
lambdaBin = (lambda > conductivities(1));
volFrac(2) = sum(lambdaBin)/length(lambdaBin);
volFrac(1) = 1 - volFrac(2);

if strcmp(hilo, 'hi')
    f = @(l) (conductivities(2) - l)*sqrt(conductivities(1)/l) - ...
        (1 - volFrac(2))*(conductivities(2) - conductivities(1));
elseif strcmp(hilo, 'lo')
    f = @(l) (conductivities(1) - l)*sqrt(conductivities(2)/l) - ...
        (1 - volFrac(1))*(conductivities(1) - conductivities(2));
else
    error('DEM for high or low conducting phase as inclusion/matrix?')
end

lambdaEff = fzero(f, [conductivities(1) conductivities(2)]);


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
    error('Which transformation for effective conductivity in differentialEffectiveMedium?')
end


end

