function [d_r] = FEMgrad(FEMout, domain, conductivity, convectionField)
%Compute derivatives of FEM equation system r = K*Y - F w.r.t. Lambda_e
%ONLY VALID FOR ISOTROPIC HEAT CONDUCTIVITY MATRIX D!!!


    function gradKK = get_glob_stiff_gradient(grad_loc_k)
        gradKK = sparse(domain.Equations(:,1), domain.Equations(:,2), grad_loc_k(domain.kIndex));
    end

if domain.useConvection
    convGradMatX = sparse(1:2:7, 1:4, ones(1, 4), 8, 4);
    convGradMatY = sparse(2:2:8, 1:4, ones(1, 4), 8, 4);
end


% (d/d Lambda_e) k^(e) = (1/Lambda_e) k^(e)     as k^(e) linear in Lambda_e
if domain.useConvection
    d_r = zeros(3*domain.nEl, domain.nEq);
else
    d_r = zeros(domain.nEl, domain.nEq);
end

for e = 1:domain.nEl
    gradLocStiffCond = zeros(4, 4, domain.nEl);
    gradLocStiffConvX = gradLocStiffCond;
    gradLocStiffConvY = gradLocStiffCond;
    gradLocStiffCond(:, :, e) = FEMout.diffusionStiffness(:, :, e)/conductivity(e);     %gradient of local stiffnesses
    
    gradK = get_glob_stiff_gradient(gradLocStiffCond);
    gradF = get_glob_force_gradient(domain, gradLocStiffCond(:, :, e), e);
    if(domain.useConvection)
        gradLocStiffConvX(:, :, e) = domain.convectionMatrix(:, :, e)*convGradMatX;
        gradLocStiffConvY(:, :, e) = domain.convectionMatrix(:, :, e)*convGradMatY;
        gradKconvX = get_glob_stiff_gradient(gradLocStiffConvX);
        gradFconvX = get_glob_force_gradient(domain, gradLocStiffConvX(:, :, e), e);
        gradKconvY = get_glob_stiff_gradient(gradLocStiffConvY);
        gradFconvY = get_glob_force_gradient(domain, gradLocStiffConvY(:, :, e), e);
        
%         gradK = [gradK; gradKconvX; gradKconvY];
%         gradF = [gradF; gradFconvX; gradFconvY];
    end
    
    d_r(e, :) = (gradK*FEMout.naturalTemperatures - gradF)';
    if domain.useConvection
        d_r(e + domain.nEl, :) = (gradKconvX*FEMout.naturalTemperatures - gradFconvX)';
        d_r(e + 2*domain.nEl, :) = (gradKconvY*FEMout.naturalTemperatures - gradFconvY)';
    end
    
    
    
    
    %Finite difference gradient check
    FDcheck = false;
    if FDcheck
        disp('Gradient check K and F')
        d = 1e-4;
        conductivityFD = conductivity;
        conductivityFD(e) = conductivityFD(e) + d;
        
        DFD = zeros(2, 2, domain.nEl);
        for j = 1:domain.nEl
            DFD(:, :, j) =  conductivityFD(j)*eye(2);
        end
        control.plt = false;
        if domain.useConvection
            FEMoutFD = heat2d(domain, DFD, convectionField);
        else
            FEMoutFD = heat2d(domain, DFD);
        end
        
        gradKFD = (FEMoutFD.globalStiffness - FEMout.globalStiffness)/d;
        e
        K = full(FEMout.globalStiffness)
        KFD = full(FEMoutFD.globalStiffness)
        gK = full(gradK)
        gKFD = full(gradKFD)
        diffGradK = full(gradK - gradKFD)
%         relgradK = full(gradKFD./gradK)
        
        gradFFD = (FEMoutFD.globalForce - FEMout.globalForce)/d
        gradF
        diffGradF = gradF - gradFFD
%         relgradF = gradFFD./gradF
        pause
    end
    
    gradK = [];
    gradF = [];
end





end

