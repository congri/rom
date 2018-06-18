function [f] = get_loc_force_v2(e, domain, kin)
%Gives local force vector
%for f_e see e.g. Hughes eq. 2.5.8

%Boundary value temperature of element e
%Tb = zeros(4, 1);
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

if Tbflag
    %fT = kin(:, :, e)*Tb;
    %f = domain.fh(:,e) + domain.fs(:,e) - fT;
    f = domain.f_tot(:, e) - kin(:, :, e)*Tb;
else
    %f = domain.fh(:,e) + domain.fs(:,e);
    f = domain.f_tot(:, e);
end
    
end