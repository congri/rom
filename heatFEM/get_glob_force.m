function [F] = get_glob_force(mesh, k)
%Assemble global force vector

neq = max(mesh.nodalCoordinates(3,:));
F = zeros(neq,1);

for e = 1:mesh.nEl
%     f = get_loc_force(e, domain, k);
%     f = get_loc_force_v2(e, domain, k);
    
    
    Tbflag = false;
    for i = 1:4
        globNode = mesh.globalNodeNumber(e, i);
        if(any(globNode == mesh.essentialNodes))
            if ~Tbflag
                Tb = zeros(4, 1);
            end
            Tb(i) = mesh.essentialTemperatures(globNode);
            Tbflag = true;
        end
    end

    for ln = 1:4
        eqn = mesh.lm(e, ln);
        if(eqn ~= 0)
            F(eqn) = F(eqn) + mesh.f_tot(ln, e);
            if Tbflag
                df = - k(:, :, e)*Tb;
                F(eqn) = F(eqn) + df(ln);
            end
        end
    end
end

end

