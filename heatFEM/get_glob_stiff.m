function [K] = get_glob_stiff(domain, k)
%Gives the global stiffness matrix

K = zeros(domain.nEq);

%Can this be done more efficiently?
%Assign global stiffness matrix
localNodeInit = 1:4;
for e = 1:domain.nEl
    equations = domain.lm(e, localNodeInit);
    localNode = localNodeInit(equations > 0)
    equations = equations(equations > 0);
%     localNode = localNodeInit(equations > 0)
equations
    K(equations, equations) = K(equations, equations) + k(localNode,localNode,e);
end

K

end

