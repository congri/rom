function [f] = get_loc_force(e, domain, kin)
%Gives local force vector
%for f_e see e.g. Hughes eq. 2.5.8

    %Contribution due to essential boundaries
    %local stiffness matrix k
    k = kin(:, :, e);
    
    %Boundary value temperature of element e
    Tb = zeros(4,1);
    Tbflag = 0;
    for i = 1:4
        globNode = domain.globalNodeNumber(e, i);
        if(any(globNode == domain.essentialNodes))
            Tb(i) = domain.essentialTemperatures(globNode);
            Tbflag = 1;
        end
    end

    if(Tbflag)
        fT = k*Tb;
        f = domain.fh(:,e) + domain.fs(:,e) - fT;
    else
        f = domain.fh(:,e) + domain.fs(:,e);
    end
    
end

