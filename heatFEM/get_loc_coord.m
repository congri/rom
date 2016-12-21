function [lc] = get_loc_coord(d)
%Gives arrays taking the element and local node number and
%giving the nodal coordinate

lc = zeros(d.nEl,4,2);
for e = 1:d.nEl
        %x coordinates
        lc(e,1,1) = (e-1)*d.lElX - floor((e - 1)/d.nElX)*d.nElX*d.lElX;
        lc(e,2,1) = e*d.lElX - floor((e - 1)/d.nElX)*d.nElX*d.lElX;
        lc(e,3,1) = e*d.lElX - floor((e - 1)/d.nElX)*d.nElX*d.lElX;
        lc(e,4,1) = (e-1)*d.lElX - floor((e - 1)/d.nElX)*d.nElX*d.lElX;

        %y coordinates
        lc(e,1,2) = floor((e - 1)/d.nElX)*d.lElY;
        lc(e,2,2) = floor((e - 1)/d.nElX)*d.lElY;
        lc(e,3,2) = floor((e - 1)/d.nElX)*d.lElY + d.lElY;
        lc(e,4,2) = floor((e - 1)/d.nElX)*d.lElY + d.lElY;
end

end

