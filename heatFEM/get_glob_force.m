function [F] = get_glob_force(domain, k)
%Assemble global force vector

neq = max(domain.nodalCoordinates(3,:));
F = zeros(neq,1);

for e = 1:domain.nEl
%     f = get_loc_force(e, domain, k);
%     f = get_loc_force_v2(e, domain, k);
    
    
    Tbflag = false;
    for i = 1:4
        globNode = domain.globalNodeNumber(e, i);
        if(any(globNode == domain.essentialNodes))
            if ~Tbflag
                Tb = zeros(4, 1);
            end
            Tb(i) = domain.essentialTemperatures(globNode);
            Tbflag = true;
        end
    end

    for ln = 1:4
        eqn = domain.lm(e, ln);
        if(eqn ~= 0)
            F(eqn) = F(eqn) + domain.f_tot(ln, e);
            if Tbflag
                df = - k(:, :, e)*Tb;
                F(eqn) = F(eqn) + df(ln);
            end
        end
    end
end

end

