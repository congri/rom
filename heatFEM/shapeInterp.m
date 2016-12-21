function [W] = shapeInterp(domainc, domainf)

%E(e) gives coarse element of fine element e
% [E] = get_coarse_el(domainf.nEl, domainc.nEl, 1:domainf.nEl);

    function [N, E] = shapeFunctionValues2(x)
        %coarse element
        row = floor(x(2)/domainc.lElY) + 1;
        %upper boundary of domain
        if row > domainc.nElY
            row = domainc.nElY;
        end
        col = floor(x(1)/domainc.lElX) + 1;
        %right boundary of domain
        if col > domainc.nElX
            col = domainc.nElX;
        end
        %E is coarse element x is in
        E = (row - 1)*(domainc.nElX) + col;
        
        %shape function values
        N(1) =(1/domainc.AEl)*(x(1) - domainc.lc(E, 2, 1))*(x(2) - domainc.lc(E, 4, 2));
        N(2,1) = -(1/domainc.AEl)*(x(1) - domainc.lc(E, 1, 1))*(x(2) - domainc.lc(E, 4, 2));
        N(3) = (1/domainc.AEl)*(x(1) - domainc.lc(E, 1, 1))*(x(2) - domainc.lc(E, 1, 2));
        N(4) = -(1/domainc.AEl)*(x(1) - domainc.lc(E, 2, 1))*(x(2) - domainc.lc(E, 1, 2));
    end

tic
R = zeros(4*domainf.nNodes, 1);
C = zeros(4*domainf.nNodes, 1);
Nvec = zeros(4*domainf.nNodes, 1);
is = 1;
ie = 4;
%r is the finescale global node number and the row index of W
for r = 1:domainf.nNodes
    %coordinate of fine node
    x(1) = domainf.nodalCoordinates(1, r);
    x(2) = domainf.nodalCoordinates(2, r);
    [N, E] = shapeFunctionValues2(x);

    %column indices of W, 4 for every local node of E
    c = domainc.globalNodeNumber(E, :);
    R(is:ie) = r;
    C(is:ie) = c;
    Nvec(is:ie) = N;
    is = is + 4;
    ie = ie + 4;
end
W = sparse(R, C, Nvec);
W_assembly_time = toc

end

