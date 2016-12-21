function [F] = get_glob_force(domain, k)
%Assemble global force vector

neq = max(domain.nodalCoordinates(3,:));
F = zeros(neq,1);

for e = 1:domain.nEl
    f = get_loc_force(e, domain, k);
    for ln = 1:4
        eqn = domain.lm(e, ln);
        if(eqn ~= 0)
            F(eqn) = F(eqn) + f(ln);
        end
    end
end

end

