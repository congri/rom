function [d_r] = FEMgrad(FEMout, domain, conductivity)
%Compute derivatives of FEM equation system r = K*Y - F w.r.t. Lambda_e
%ONLY VALID FOR ISOTROPIC HEAT CONDUCTIVITY MATRIX D!!!


    function gradKK = get_glob_stiff_gradient(grad_loc_k)
        gradKK = sparse(domain.Equations(:,1), domain.Equations(:,2), grad_loc_k(domain.kIndex));
    end



% (d/d Lambda_e) k^(e) = (1/Lambda_e) k^(e)     as k^(e) linear in Lambda_e
d_r = zeros(domain.nEl, domain.nEq);
for e = 1:domain.nEl
    gradLocStiff = zeros(4, 4, domain.nEl);
    gradLocStiff(:, :, e) = FEMout.localStiffness(:, :, e)/conductivity(e);     %gradient of local stiffnesses
    gradK = get_glob_stiff_gradient(gradLocStiff);
    gradF = get_glob_force_gradient(domain, gradLocStiff(:, :, e), e);
    
    d_r(e, :) = (gradK*FEMout.naturalTemperatures - gradF)';
    
    
    
    
    
    %Finite difference gradient check
    FDcheck = false;
    if FDcheck
        disp('Gradient check K and F')
        d = 1e-4;
        FDgrad = zeros(domain.nEl, 1);
        conductivityFD = conductivity;
        conductivityFD(e) = conductivityFD(e) + d;
        
        DFD = zeros(2, 2, domain.nEl);
        for j = 1:domain.nEl
            DFD(:, :, j) =  conductivityFD(j)*eye(2);
        end
        control.plt = false;
        FEMoutFD = heat2d(domain, physical, control, DFD);
        
        gradKFD = (FEMoutFD.globalStiffness - FEMout.globalStiffness)/d;
%         gradK
        relgradK = gradKFD./gradK
        
        gradFFD = (FEMoutFD.globalForce - FEMout.globalForce)/d
        gradF
        relgradF = gradFFD./gradF
        pause
    end
end





end

