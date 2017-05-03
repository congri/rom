function [lc] = get_loc_coord(d)
%Gives arrays taking the element and local node number and
%giving the nodal coordinate

lc = zeros(d.nEl,4,2);
for e = 1:d.nEl
    row = floor((e - 1)/d.nElX) + 1;
    col = mod((e - 1), d.nElX) + 1;
    
    %this is for a regular rectangular mesh
%     %x coordinates
%     lc(e,1,1) = (e-1)*d.lElX(e) - floor((e - 1)/d.nElX)*d.nElX*d.lElX(e);
%     lc(e,2,1) = e*d.lElX(e) - floor((e - 1)/d.nElX)*d.nElX*d.lElX(e);
%     lc(e,3,1) = e*d.lElX(e) - floor((e - 1)/d.nElX)*d.nElX*d.lElX(e);
%     lc(e,4,1) = (e-1)*d.lElX(e) - floor((e - 1)/d.nElX)*d.nElX*d.lElX(e);
%     
%     %y coordinates
%     lc(e,1,2) = floor((e - 1)/d.nElX)*d.lElY(e);
%     lc(e,2,2) = floor((e - 1)/d.nElX)*d.lElY(e);
%     lc(e,3,2) = floor((e - 1)/d.nElX)*d.lElY(e) + d.lElY(e);
%     lc(e,4,2) = floor((e - 1)/d.nElX)*d.lElY(e) + d.lElY(e);

    %x-coordinates
    lc(e, 1, 1) = d.cum_lElX(col);
    lc(e, 2, 1) = d.cum_lElX(col + 1);
    lc(e, 3, 1) = lc(e, 2, 1);
    lc(e, 4, 1) = lc(e, 1, 1);
    
    %y-coordinates
    lc(e, 1, 2) = d.cum_lElY(row);
    lc(e, 2, 2) = lc(e, 1, 2);
    lc(e, 3, 2) = d.cum_lElY(row + 1);
    lc(e, 4, 2) = lc(e, 3, 2);
end

end

