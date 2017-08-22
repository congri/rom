function [convection] = convectionField(x, coeff)
%Set up incompressible convection field here
convection = [coeff(1) + coeff(2)*x(1) + coeff(3)*x(2);...
    coeff(4) - coeff(2)*x(2) + coeff(5)*x(1)];

end

