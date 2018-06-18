function [Out] = heat2d_v2(mesh, D)
%2D heat conduction main function
%Gives back temperature on point x

%Compute local stiffness matrices, once and for all
Out.diffusionStiffness = zeros(4, 4, mesh.nEl);

for e = 1:mesh.nEl
    Out.diffusionStiffness(:, :, e) = D(e)*mesh.d_loc_stiff(:, :, e);
end

% %Global stiffness matrix
% localStiffness = Out.diffusionStiffness;

Out.globalStiffness = get_glob_stiff2(mesh, Out.diffusionStiffness);
%Global force vector
Out.globalForce = get_glob_force(mesh, Out.diffusionStiffness);

%Finally solving the equation system
Out.naturalTemperatures = Out.globalStiffness\Out.globalForce;


%Temperature field
Tf = zeros(mesh.nNodes, 1);
Tf(mesh.id) = Out.naturalTemperatures;
Tff = zeros(mesh.nElX + 1, mesh.nElY + 1);

for i = 1:mesh.nNodes
    Tff(i) = Tf(i);
    if(any(i == mesh.essentialNodes))
        %node i is essential
        Tff(i) = mesh.essentialTemperatures(i);
    end
end

Tff = Tff';
Out.Tff = Tff;


end