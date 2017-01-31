function [out] = weakContrastExpansion(lambda, conductivities, transform, hilo)
%Weak contrast expansion Torquato eq. 20.77

%Binary conductivity vector
lambdaBin = (lambda > conductivities(1));
volFrac(2) = sum(lambdaBin)/length(lambdaBin);
volFrac(1) = 1 - volFrac(2);

if strcmp(hilo, 'hi')
    %Torquato 20.76, 20.78 - 20.81
    a1 = volFrac(1);
    a2 = -.5*volFrac(1)*volFrac(2);
    a3 = .25*volFrac(1)*volFrac(2)*(volFrac(2) + volFrac(1));
    a4 = .125*volFrac(1)*volFrac(2)*...
        (-volFrac(2)^2 - (1 + 2*volFrac(2))*volFrac(1));
    
    %Torquato 20.77
    condDiff = conductivities(1) - conductivities(2);
    lambdaEff = conductivities(2) + a1*condDiff + a2*((condDiff^2)/(conductivities(2))) + ...
        a3*((condDiff^3)/conductivities(2)^2) + a4*((condDiff^4)/conductivities(2)^3);
elseif strcmp(hilo, 'lo')
    %Torquato 20.76, 20.78 - 20.81
    a1 = volFrac(2);
    a2 = -.5*volFrac(1)*volFrac(2);
    a3 = .25*volFrac(1)*volFrac(2)*(volFrac(2) + volFrac(1));
    a4 = .125*volFrac(1)*volFrac(2)*...
        (-volFrac(1)^2 - (1 + 2*volFrac(1))*volFrac(2));
    
    %Torquato 20.77
    condDiff = conductivities(2) - conductivities(1);
%     l1 = conductivities(1) + a1*condDiff
%     l2 = conductivities(1) + a1*condDiff + a2*((condDiff^2)/(conductivities(1)))
%     l3 = conductivities(1) + a1*condDiff + a2*((condDiff^2)/(conductivities(1))) + ...
%         a3*((condDiff^3)/conductivities(1)^2)
    lambdaEff = conductivities(1) + a1*condDiff + a2*((condDiff^2)/(conductivities(1))) + ...
        a3*((condDiff^3)/conductivities(1)^2) + a4*((condDiff^4)/conductivities(1)^3);
else
    error('Phase assignment in weakContrastExpansion')
end

if strcmp(transform, 'log')
    if lambdaEff <= 0
        error('You cannot use log-transform for lambdaEff <= 0')
    end
    out = log(lambdaEff);
elseif strcmp(transform, 'logit')
    %Limitation of effective conductivity
    if lambdaEff <= 0
        error('You cannot use logit-transform for lambdaEff <= 0')
    end
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

