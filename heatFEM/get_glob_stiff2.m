function [K] = get_glob_stiff2(domain, k)
%Gives global stiffness matrix K

K = sparse(domain.Equations(:,1), domain.Equations(:,2), k(domain.kIndex));

end

