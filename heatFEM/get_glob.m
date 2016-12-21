function [g] = get_glob(d)
%Get global node number from global element number
%and local node number

g = zeros(d.nEl, 4, 'int32');
for e = 1:d.nEl
    for l = 1:4
        g(e,1) = e + floor((e - 1)/d.nElX);
        g(e,2) = e + floor((e - 1)/d.nElX) + 1;
        g(e,3) = g(e,1) + d.nElX + 2;
        g(e,4) = g(e,1) + d.nElX + 1;
    end
end
    
end

