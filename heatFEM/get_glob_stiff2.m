function [K] = get_glob_stiff2(mesh, k)
%Gives global stiffness matrix K

K = sparse(mesh.Equations(:,1), mesh.Equations(:,2), k(mesh.kIndex));


end

