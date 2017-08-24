function [convection] = convectionField(x, coeff)
%Set up incompressible convection field here

mode = 'sin';
if strcmp(mode, 'linear')
    convection = [coeff(1) + coeff(2)*x(1) + coeff(3)*x(2);...
        coeff(4) - coeff(2)*x(2) + coeff(5)*x(1)];
elseif strcmp(mode, 'sin')
    amplitude = 5;
    convection = amplitude*[sum(sin(coeff*x(1)).*sin(coeff*x(2)));...
        sum(cos(coeff*x(1)).*cos(coeff*x(2)))];
end

end

