function [d_r] = FEMgrad(FEMout, mesh, conductivity)
%Compute derivatives of FEM equation system r = K*Y - F w.r.t. Lambda_e
%ONLY VALID FOR ISOTROPIC HEAT CONDUCTIVITY MATRIX D!!!


%     function gradKK = get_glob_stiff_gradient(grad_loc_k)
%         gradKK = sparse(mesh.Equations(:,1),...
%             mesh.Equations(:,2), grad_loc_k(mesh.kIndex));
%     end


% (d/d Lambda_e) k^(e) = (1/Lambda_e) k^(e)     as k^(e) linear in Lambda_e
d_r = zeros(mesh.nEl, mesh.nEq);

for e = 1:mesh.nEl
%     gradLocStiffCond = zeros(4, 4, mesh.nEl);
%     %gradient of local stiffnesses
%     gradLocStiffCond(:, :, e) =...
%         FEMout.diffusionStiffness(:, :, e)/conductivity(e);
%     
%     gradK = get_glob_stiff_gradient(gradLocStiffCond);
%     gradF = get_glob_force_gradient(mesh, gradLocStiffCond(:, :, e), e);
%     
%     d_r(e, :) = (gradK*FEMout.naturalTemperatures - gradF)';
    
    d_r(e, :) = (mesh.d_glob_stiff{e}*FEMout.naturalTemperatures -...
        mesh.d_glob_force{e})';    
    
    
    %Finite difference gradient check
    FDcheck = false;
    if FDcheck
        disp('Gradient check K and F')
        d = 1e-4;
        conductivityFD = conductivity;
        conductivityFD(e) = conductivityFD(e) + d;
        
        DFD = zeros(2, 2, mesh.nEl);
        for j = 1:mesh.nEl
            DFD(:, :, j) =  conductivityFD(j)*eye(2);
        end
        control.plt = false;
        FEMoutFD = heat2d(mesh, DFD);
        
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

