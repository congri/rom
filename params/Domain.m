classdef Domain
    %class describing the finite element domain

    properties (SetAccess = public)
        
        nElX                        %number of finite elements in each direction; not tested for nElX ~= nElY
        nElY
        nEl                         %total number of elements
        nNodes                      %total number of nodes
        boundaryNodes               %Nodes on the domain boundary
        essentialNodes              %essential boundary nodes
        essentialTemperatures       %Nodal temperature of essential nodes; NaN if natural
        naturalNodes                %natural boundary nodes
        boundaryElements            %Elements on boundary, counterclockwise counted
        naturalBoundaries           %nEl x 4 array holding natural boundary edges of elements
        boundaryType                %true for essential, false for natural boundary node
        lx = 1;                     %domain size; not tested for lx, ly ~= 1
        ly = 1;
        lElX                        %element size
        lElY
        AEl
        nEq                         %number of equations
        lc                          %lc gives node coordinates, taking in element number and local node number
        nodalCoordinates            %Nodal coordiante array
                                    %holds global nodal coordinates in the first two lines (x and y).
                                    %In the thrid line, the equation number is stored
        globalNodeNumber            %globalNodeNumber holds the global node number, given the element number as row
                                    %and the local node number as column indices
        Bvec                        %Shape function gradient array, precomputed for performance
        
        essentialBoundary           %essential boundary (yes or no) given local node and element number
        lm                          %lm takes element number as row and local node number as column index
                                    %and gives equation number
        id                          %Get mapping from equation number back to global node number
        Equations                   %eq. number and loc. node number precomputation for sparse stiffness assembly
        LocalNode
        kIndex
        
        fs                          %local forces due to heat source
        fh                          %local forces due to natural boundary
  
    end
    
    methods
        
        function domainObj = Domain(nElX, nElY)
            %constructor
            if nargin > 0
                domainObj.nElX = nElX;
                if nargin > 1
                    domainObj.nElY = nElY;
                    if nargin > 2
                        domainObj.lx = lx;
                        if nargin == 4
                            domainObj.ly = ly;
                        end
                    end
                end
                if nargin > 4
                    error('Wrong number of input arguments')
                end
            end
            domainObj.lElX = domainObj.lx/domainObj.nElX;
            domainObj.lElY = domainObj.ly/domainObj.nElY;
            domainObj.nEl = domainObj.nElX*domainObj.nElY;
            domainObj.nNodes = (domainObj.nElX + 1)*(domainObj.nElY + 1);
            domainObj.AEl = domainObj.lElX*domainObj.lElY;
            domainObj.boundaryNodes = int32([1:(domainObj.nElX + 1),...
                2*(domainObj.nElX + 1):(domainObj.nElX + 1):(domainObj.nElX + 1)*(domainObj.nElY + 1),...
                ((domainObj.nElX + 1)*(domainObj.nElY + 1) - 1):(-1):((domainObj.nElX + 1)*domainObj.nElY + 1),...
                (domainObj.nElX + 1)*((domainObj.nElY - 1):(-1):1) + 1]);
            domainObj.boundaryElements = int32([1:domainObj.nElX,...
                2*(domainObj.nElX):(domainObj.nElX):(domainObj.nElX*domainObj.nElY),...
                ((domainObj.nElX)*(domainObj.nElY) - 1):(-1):(domainObj.nElX*(domainObj.nElY - 1) + 1),...
                (domainObj.nElX)*((domainObj.nElY - 2):(-1):1) + 1]);
            
            %local coordinate array. First index is element number, 2 is local node, 3 is x or y
            domainObj.lc = get_loc_coord(domainObj);
            domainObj.globalNodeNumber = get_glob(domainObj);
            
            domainObj = setHeatSource(domainObj, zeros(domainObj.nEl, 1));  %zero as default

        end
        
        function domainObj = setNodalCoordinates(domainObj)
            domainObj.nodalCoordinates = get_coord(domainObj);
            domainObj.lm = domainObj.globalNodeNumber;
            Sg = size(domainObj.globalNodeNumber);
            for i = 1:Sg(1)
                for j = 1:Sg(2)
                    domainObj.lm(i,j) = domainObj.nodalCoordinates(3,domainObj.globalNodeNumber(i,j));
                end
            end
            domainObj.id = get_id(domainObj.nodalCoordinates);
            [domainObj.Equations, domainObj.LocalNode] = get_equations(domainObj.nEl, domainObj.lm);
            domainObj.Equations = double(domainObj.Equations);
            domainObj.kIndex = sub2ind([4 4 domainObj.nEl], domainObj.LocalNode(:,1),...
                domainObj.LocalNode(:,2), domainObj.LocalNode(:,3));
        end
        
        function domainObj = setBvec(domainObj)
            domainObj.nEq = max(domainObj.nodalCoordinates(3,:));
            %Gauss points
            xi1 = -1/sqrt(3);
            xi2 = 1/sqrt(3);
            
            domainObj.Bvec = zeros(8, 4, domainObj.nEl);
            for e = 1:domainObj.nEl
                for i = 1:4
                    domainObj.essentialBoundary(i, e) =...
                        ~isnan(domainObj.essentialTemperatures(domainObj.globalNodeNumber(e, i)));
                end
                %short hand notation
                x1 = domainObj.lc(e,1,1);
                x2 = domainObj.lc(e,2,1);
                y1 = domainObj.lc(e,1,2);
                y4 = domainObj.lc(e,4,2);
                
                %Coordinate transformation
                xI = 0.5*(x1 + x2) + 0.5*xi1*(x2 - x1);
                xII = 0.5*(x1 + x2) + 0.5*xi2*(x2 - x1);
                yI = 0.5*(y1 + y4) + 0.5*xi1*(y4 - y1);
                yII = 0.5*(y1 + y4) + 0.5*xi2*(y4 - y1);
                
                %Assuming bilinear shape functions here!!!
                B1 = [yI-y4 y4-yI yI-y1 y1-yI; xI-x2 x1-xI xI-x1 x2-xI];
                B2 = [yII-y4 y4-yII yII-y1 y1-yII; xII-x2 x1-xII xII-x1 x2-xII];
                %Do not forget cross terms
                B3 = [yI-y4 y4-yI yI-y1 y1-yI; xII-x2 x1-xII xII-x1 x2-xII];
                B4 = [yII-y4 y4-yII yII-y1 y1-yII; xI-x2 x1-xI xI-x1 x2-xI];
                
                %Note:in Gauss quadrature, the differential transforms as dx = (l_x/2) d xi. Hence
                %we take the additional factor of sqrt(A)/2 onto B
                domainObj.Bvec(:,:,e) = (1/(2*sqrt(domainObj.AEl)))*[B1; B2; B3; B4];
            end
        end
        
        function domainObj = setHeatSource(domainObj, heatSourceField)
            domainObj.fs = get_heat_source(heatSourceField, domainObj);
        end
        
        function domainObj = setBoundaries(domainObj, natNodes, Tb, qb)    
            %natNodes holds natural nodes counted counterclockwise around domain, starting in lower
            %left corner. Tb and qb are function handles to temperature and heat flux boundary
            %functions
            domainObj.boundaryType = true(1, 2*domainObj.nElX + 2*domainObj.nElY);
            domainObj.boundaryType(natNodes) = false;
            domainObj.essentialNodes = domainObj.boundaryNodes(domainObj.boundaryType);
            domainObj.naturalNodes = int32(domainObj.boundaryNodes(~domainObj.boundaryType));
            
            %Set essential temperatures
            domainObj.essentialTemperatures = NaN*ones(1, domainObj.nNodes);
            boundaryCoordinates = [0:domainObj.lElX:1, ones(1, domainObj.nElY),...
                (1 - domainObj.lElX):(-domainObj.lElX):0, zeros(1, domainObj.nElY - 1);...
                zeros(1, domainObj.nElX + 1), domainObj.lElY:domainObj.lElY:1,...
                ones(1, domainObj.nElX), (1 - domainObj.lElY):(-domainObj.lElY):domainObj.lElY];
            Tess = zeros(1, domainObj.nNodes);
            for i = 1:(2*domainObj.nElX + 2*domainObj.nElY)
                Tess(i) = Tb(boundaryCoordinates(:, i));
            end
            domainObj.essentialTemperatures(domainObj.essentialNodes) = Tess(domainObj.boundaryType);
            
            %Natural boundaries have to enclose natural nodes
            domainObj.naturalBoundaries = false(domainObj.nEl, 4);
            globNatNodes = domainObj.boundaryNodes(natNodes);   %global node numbers of natural nodes
            
            %Set natural boundaries
            for i = 1:numel(globNatNodes)
                %find elements containing these nodes
                natElem = find(globNatNodes(i) == domainObj.globalNodeNumber);
                [elem, ~] = ind2sub(size(domainObj.globalNodeNumber), natElem);
                %find out side of boundary (lo, r, u, le)
                if(globNatNodes(i) == 1)
                    %lower left corner
                    assert(numel(elem) == 1, 'Error: corner node in more than one element?')
                    domainObj.naturalBoundaries(1, 1) = true;
                    domainObj.naturalBoundaries(1, 4) = true;
                elseif(globNatNodes(i) == domainObj.nElX + 1)
                    %lower right corner
                    assert(numel(elem) == 1, 'Error: corner node in more than one element?')
                    domainObj.naturalBoundaries(elem, 1) = true;
                    domainObj.naturalBoundaries(elem, 2) = true;
                elseif(globNatNodes(i) == (domainObj.nElX + 1)*(domainObj.nElY + 1))
                    %upper right corner
                    assert(numel(elem) == 1, 'Error: corner node in more than one element?')
                    domainObj.naturalBoundaries(elem, 2) = true;
                    domainObj.naturalBoundaries(elem, 3) = true;
                elseif(globNatNodes(i) == (domainObj.nElX + 1)*(domainObj.nElY) + 1)
                    %upper left corner
                    assert(numel(elem) == 1, 'Error: corner node in more than one element?')
                    domainObj.naturalBoundaries(elem, 3) = true;
                    domainObj.naturalBoundaries(elem, 4) = true;
                elseif(globNatNodes(i) > 1 && globNatNodes(i) < domainObj.nElX + 1)
                    %exclusively on lower bound
                    assert(numel(elem) == 2, 'Error: boundary node not in 2 elements?')
                    domainObj.naturalBoundaries(elem(1), 1) = true;
                    domainObj.naturalBoundaries(elem(2), 1) = true;
                elseif(mod(globNatNodes(i), domainObj.nElX + 1) == 0)
                    %exclusively on right bound
                    assert(numel(elem) == 2, 'Error: boundary node not in 2 elements?')
                    domainObj.naturalBoundaries(elem(1), 2) = true;
                    domainObj.naturalBoundaries(elem(2), 2) = true;
                elseif(globNatNodes(i) > (domainObj.nElX + 1)*(domainObj.nElY) + 1)
                    %exclusively on upper bound
                    assert(numel(elem) == 2, 'Error: boundary node not in 2 elements?')
                    domainObj.naturalBoundaries(elem(1), 3) = true;
                    domainObj.naturalBoundaries(elem(2), 3) = true;
                elseif(mod(globNatNodes(i), domainObj.nElX + 1) == 1)
                    %exclusively on left bound
                    assert(numel(elem) == 2, 'Error: boundary node not in 2 elements?')
                    domainObj.naturalBoundaries(elem(1), 4) = true;
                    domainObj.naturalBoundaries(elem(2), 4) = true;
                end
            end
            
            %Finally set local forces due to natural boundaries
            domainObj.fh = get_flux_force(domainObj, qb);
            domainObj = setNodalCoordinates(domainObj);
            domainObj = setBvec(domainObj);
        end
        
        function domainObj = shrink(domainObj)
            %To save memory. We use that on finescale domain to save memory
            domainObj.lc = [];
            domainObj.Equations = [];
            domainObj.kIndex = [];
            domainObj.boundaryNodes = [];
            domainObj.essentialNodes = [];
            domainObj.essentialTemperatures = [];
            domainObj.naturalNodes = [];
            domainObj.boundaryElements = [];
            domainObj.naturalBoundaries = [];
            domainObj.boundaryType = [];
            domainObj.lx = [];
            domainObj.ly = [];
            domainObj.AEl = [];
            domainObj.nEq = [];
            domainObj.nodalCoordinates = [];
            domainObj.globalNodeNumber =[];
            domainObj.Bvec = [];
            domainObj.essentialBoundary = [];
            domainObj.lm = [];
            domainObj.id = [];
            domainObj.LocalNode = [];
            domainObj.fs = [];
            domainObj.fh = [];
        end
    end
    
end
























