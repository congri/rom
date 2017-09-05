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
        lElX                        %element length in X
        lElY                        %element length in Y
        cum_lElX                    %grid vectors of FEM mesh
        cum_lElY                    
        AEl                         %Element surface
        nEq                         %number of equations
        lc                          %lc gives node coordinates, taking in element number and local node number
        nodalCoordinates            %Nodal coordiante array
                                    %holds global nodal coordinates in the first two lines (x and y).
                                    %In the thrid line, the equation number is stored
        globalNodeNumber            %globalNodeNumber holds the global node number, given the element number as row
                                    %and the local node number as column indices
        Bvec                        %Shape function gradient array, precomputed for performance
        d_N                         %Shape function gradient array for Gauss quadrature of convection matrix
        NArray
        convectionMatrix            %Precomputed matrix to accelerate convection term integration
        useConvection               %Use a convection term in the PDE
        
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
        function domainObj = Domain(nElX, nElY, lElX, lElY)
            %constructor
            %nElX andnElY are number of elements in x- and y-direction
            %lElX, lElY are vectors specifying element lengths in x- and y-directions. i-th element
            %in lElX: i-th column; j-th element in lElY: j-th row
            if nargin > 0
                domainObj.nElX = nElX;
                if nargin > 1
                    domainObj.nElY = nElY;
                end
            end
            %Set square mesh as default
            if nargin < 3
                lElX = domainObj.lx/domainObj.nElX*ones(1, domainObj.nElX);
                lElY = domainObj.ly/domainObj.nElY*ones(1, domainObj.nElY);
            end
            domainObj.nEl = domainObj.nElX*domainObj.nElY;
            assert(numel(lElX) == domainObj.nElX, 'incorrect number of elements specified in element length vector')
            assert(numel(lElY) == domainObj.nElY, 'incorrect number of elements specified in element length vector')
            diffX = abs(sum(lElX) - domainObj.lx);
            diffY = abs(sum(lElY) - domainObj.ly);
            assert(diffX < eps, 'element lengths do not sum up to lx')
            assert(diffY < eps, 'element lengths do not sum up to ly')
            domainObj.lElX = zeros(1, domainObj.nEl);
            domainObj.lElY = zeros(1, domainObj.nEl);
            domainObj.AEl = zeros(1, domainObj.nEl);
            for e = 1:domainObj.nEl
                domainObj.lElX(e) = lElX(mod((e - 1), domainObj.nElX) + 1);
                domainObj.lElY(e) = lElY(floor((e - 1)/domainObj.nElX) + 1);
                domainObj.AEl(e) = domainObj.lElX(e)*domainObj.lElY(e);
            end
            domainObj.cum_lElX = cumsum([0 lElX]);
            domainObj.cum_lElY = cumsum([0 lElY]);
            domainObj.nNodes = (domainObj.nElX + 1)*(domainObj.nElY + 1);
            domainObj.boundaryNodes = int32([1:(domainObj.nElX + 1),...
                2*(domainObj.nElX + 1):(domainObj.nElX + 1):(domainObj.nElX + 1)*(domainObj.nElY + 1),...
                ((domainObj.nElX + 1)*(domainObj.nElY + 1) - 1):(-1):((domainObj.nElX + 1)*domainObj.nElY + 1),...
                (domainObj.nElX + 1)*((domainObj.nElY - 1):(-1):1) + 1]);
            domainObj.boundaryElements = int32([1:domainObj.nElX,...
                2*(domainObj.nElX):(domainObj.nElX):(domainObj.nElX*domainObj.nElY),...
                ((domainObj.nElX)*(domainObj.nElY) - 1):(-1):(domainObj.nElX*(domainObj.nElY - 1) + 1),...
                (domainObj.nElX)*((domainObj.nElY - 2):(-1):1) + 1]);
            
            %local coordinate array. First index is element number, 2 is local node, 3 is x or y
            domainObj = domainObj.setLocCoord;
            domainObj = domainObj.setGlobalNodeNumber;
            
            domainObj = setHeatSource(domainObj, zeros(domainObj.nEl, 1));  %zero as default
        end

        function domainObj = setLocCoord(domainObj)
            %Gives arrays taking the element and local node number and giving the nodal coordinate
            
            domainObj.lc = zeros(domainObj.nEl, 4, 2);
            for e = 1:domainObj.nEl
                row = floor((e - 1)/domainObj.nElX) + 1;
                col = mod((e - 1), domainObj.nElX) + 1;

                %x-coordinates
                domainObj.lc(e, 1, 1) = domainObj.cum_lElX(col);
                domainObj.lc(e, 2, 1) = domainObj.cum_lElX(col + 1);
                domainObj.lc(e, 3, 1) = domainObj.lc(e, 2, 1);
                domainObj.lc(e, 4, 1) = domainObj.lc(e, 1, 1);
                
                %y-coordinates
                domainObj.lc(e, 1, 2) = domainObj.cum_lElY(row);
                domainObj.lc(e, 2, 2) = domainObj.lc(e, 1, 2);
                domainObj.lc(e, 3, 2) = domainObj.cum_lElY(row + 1);
                domainObj.lc(e, 4, 2) = domainObj.lc(e, 3, 2);
            end
        end

        function domainObj = setGlobalNodeNumber(domainObj)
            %Get global node number from global element number and local node number
            
            domainObj.globalNodeNumber = zeros(domainObj.nEl, 4, 'int32');
            for e = 1:domainObj.nEl
                for l = 1:4
                    domainObj.globalNodeNumber(e,1) = e + floor((e - 1)/domainObj.nElX);
                    domainObj.globalNodeNumber(e,2) = e + floor((e - 1)/domainObj.nElX) + 1;
                    domainObj.globalNodeNumber(e,3) = domainObj.globalNodeNumber(e,1) + domainObj.nElX + 2;
                    domainObj.globalNodeNumber(e,4) = domainObj.globalNodeNumber(e,1) + domainObj.nElX + 1;
                end
            end
        end

        function domainObj = setId(domainObj)
            %put in equation number, get back global node number
            
            [eqs, i] = sort(domainObj.nodalCoordinates(3, :));
            
            domainObj.id = [eqs', i'];
            
            init = find(eqs == 1);
            
            domainObj.id(1:(init-1), :) = [];
            
            domainObj.id = domainObj.id(:, 2);
            domainObj.id = uint32(domainObj.id);
        end

        function domainObj = getEquations(domainObj)
            %Equation number array for sparse global stiffness assembly
            
            localNodeInit = 1:4;
            %preallocate
            domainObj.Equations = zeros(16*domainObj.nEl, 2);
            domainObj.LocalNode = zeros(16*domainObj.nEl, 3);
            eq = 0; %equation number index
            for e = 1:domainObj.nEl
                equationslm = domainObj.lm(e, localNodeInit);
                equations = equationslm(equationslm > 0);
                localNode = localNodeInit(equationslm > 0);
                prevnEq = eq;
                eq = eq + numel(equations)^2;
                
                [Equations1, Equations2] = meshgrid(equations);
                domainObj.Equations((prevnEq + 1):eq, :) = [Equations1(:) Equations2(:)];
                
                [LocalNode1, LocalNode2] = meshgrid(localNode);
                domainObj.LocalNode((prevnEq + 1):eq, :) =...
                   [LocalNode1(:) LocalNode2(:) repmat(e, length(equations)^2, 1)];
            end
            
            %Shrink to fit
            domainObj.Equations((eq + 1):end, :) = [];
            domainObj.LocalNode((eq + 1):end, :) = [];
        end

        function domainObj = getCoord(domainObj)
            %Gives nodal coordinates in the first two rows and equation number from
            %global node number in the third row. Temperature of essential boundaries
            %is given in the fourth row, heat flux on natural boundaries in the fifth
            %Assign global node coordinates and equation numbers
            %In clockwise direction, the first node of every side is considered to belong to the boundary. The
            %last node is considered to belong to the next boundary. E.g. on a grid 5x5 nodes, nodes 1 - 4
            %belong to the lower boundary, nodes 5, 10, 15, 20 to the right, nodes 25, 24, 23, 22 to the upper
            %and 21, 16, 11, 6 to the left boundary.

            j = 1;  %equation number index
            domainObj.nodalCoordinates = NaN*zeros(3, domainObj.nNodes);
            for i = 1:domainObj.nNodes
                row = floor((i - 1)/(domainObj.nElX + 1)) + 1;
                col = mod((i - 1), (domainObj.nElX + 1)) + 1;
                x = domainObj.cum_lElX(col);
                y = domainObj.cum_lElY(row);
                domainObj.nodalCoordinates(1, i) = x;
                domainObj.nodalCoordinates(2, i) = y;
                
                if(any(domainObj.essentialNodes == i))
                    %essential node, no equation number assigned
                    domainObj.nodalCoordinates(3, i) = 0;
                else
                    %Assign equation number j
                    domainObj.nodalCoordinates(3, i) = j;
                    j = j + 1;
                end
            end 
        end

        function domainObj = setNodalCoordinates(domainObj)
            domainObj = getCoord(domainObj);
            domainObj.lm = domainObj.globalNodeNumber;
            for i = 1:size(domainObj.globalNodeNumber, 1)
                for j = 1:size(domainObj.globalNodeNumber, 2)
                    domainObj.lm(i, j) = domainObj.nodalCoordinates(3, domainObj.globalNodeNumber(i, j));
                end
            end
            domainObj = setId(domainObj);
            domainObj = getEquations(domainObj);
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
                domainObj.Bvec(:, :, e) = (1/(2*sqrt(domainObj.AEl(e))))*[B1; B2; B3; B4];
            end
        end
        
        function domainObj = setConvectionMatrix(domainObj)
            %Only call if necessary. This is memory consuming!
            disp('Setting convection matrix...')
            
            domainObj = domainObj.elementShapeFunctionArray;
            domainObj = domainObj.elementShapeFunctionGradients;
            domainObj.convectionMatrix = zeros(4, 8, domainObj.nEl);
            for e = 1:domainObj.nEl
                domainObj.convectionMatrix(:, :, e) = domainObj.NArray(:, :, e)*domainObj.d_N(:, :, e);
            end
            disp('done')
        end

        function domainObj = setHeatSource(domainObj, heatSourceField)
            %Gets the elements of the local force due to the heat source (an array with
            %input element number e and local node number i
            
            %Gauss points
            xi1 = -1/sqrt(3);
            xi2 = 1/sqrt(3);
            eta1 = -1/sqrt(3);
            eta2 = 1/sqrt(3);
            
            domainObj.fs = zeros(4, domainObj.nEl);

            for e = 1:domainObj.nEl
                %short hand notation. Coordinates of local nodes
                x1 = domainObj.lc(e, 1, 1);
                x2 = domainObj.lc(e, 2, 1);
                y1 = domainObj.lc(e, 1, 2);
                y4 = domainObj.lc(e, 4, 2);
                
                %Coordinate transformation
                xI = 0.5*(x1 + x2) + 0.5*xi1*(x2 - x1);
                xII = 0.5*(x1 + x2) + 0.5*xi2*(x2 - x1);
                yI = 0.5*(y1 + y4) + 0.5*eta1*(y4 - y1);
                yII = 0.5*(y1 + y4) + 0.5*eta2*(y4 - y1);
                
                
                domainObj.fs(1, e) = heatSourceField(e)*(1/domainObj.AEl(e))*((xI - x2)*...
                    (yI - y4) + (xII - x2)*(yII - y4) + (xI - x2)*(yII - y4) + (xII - x2)*(yI - y4));
                domainObj.fs(2, e) = -heatSourceField(e)*(1/domainObj.AEl(e))*((xI - x1)*...
                    (yI - y4) + (xII - x1)*(yII - y4) + (xI - x1)*(yII - y4) + (xII - x1)*(yI - y4));
                domainObj.fs(3, e) = heatSourceField(e)*(1/domainObj.AEl(e))*((xI - x1)*...
                    (yI - y1) + (xII - x1)*(yII - y1) + (xI - x1)*(yII - y1) + (xII - x1)*(yI - y1));
                domainObj.fs(4, e) = -heatSourceField(e)*(1/domainObj.AEl(e))*((xI - x2)*...
                    (yI - y1) + (xII - x2)*(yII - y1) + (xI - x2)*(yII - y1) + (xII - x2)*(yI - y1));
            end
        end
        
        function N = elementShapeFunctions(domainObj, x, y, xe, Ael, component)
            %Gives values of element shape functions
            %   x, y:  domain variables
            %   xe: xe(1) = x_1^e, xe(2) = x_2^e, xe(3) = y_1^e, xe(4) = y_4^e, see Fish&Belytschko p163
            if(nargin < 6)
                N = zeros(4, 1);
                N(1) = (x - xe(2)).*(y - xe(4));
                N(2) = -(x - xe(1)).*(y - xe(4));
                N(3) = (x - xe(1)).*(y - xe(3));
                N(4) = -(x - xe(2)).*(y - xe(3));
                N = N/Ael;
            else
                switch component
                    case 1
                        N = (x - xe(2)).*(y - xe(4));
                    case 2
                        N = -(x - xe(1)).*(y - xe(4));
                    case 3
                        N = (x - xe(1)).*(y - xe(3));
                    case 4
                        N = -(x - xe(2)).*(y - xe(3));
                    otherwise
                        error('Which local node?')
                end
                N = N/Ael;
            end
        end
        
        function domainObj = elementShapeFunctionGradients(domainObj)
            %Gives values of element shape function gradient arrays for Gauss quadrature
            %of convection matrix
            %This is similar to Bvec, but with different array arrangement
            
            %Gauss points
            xi1 = -1/sqrt(3);
            xi2 = 1/sqrt(3);
            domainObj.d_N = zeros(4, 8, domainObj.nEl);
            for e = 1:domainObj.nEl
                %short hand notation
                x1 = domainObj.lc(e, 1, 1);
                x2 = domainObj.lc(e, 2, 1);
                y1 = domainObj.lc(e, 1, 2);
                y4 = domainObj.lc(e, 4, 2);
                
                %Coordinate transformation of Gauss quadrature points xi1 and xi2
                xI = 0.5*(x1 + x2) + 0.5*xi1*(x2 - x1);
                xII = 0.5*(x1 + x2) + 0.5*xi2*(x2 - x1);
                yI = 0.5*(y1 + y4) + 0.5*xi1*(y4 - y1);
                yII = 0.5*(y1 + y4) + 0.5*xi2*(y4 - y1);
                
                %Assuming bilinear shape functions here!!!
                B = [yI - y4, yI - y4, yII - y4, yII - y4;...
                    xI - x2, xII - x2, xI - x2, xII - x2;...
                    y4 - yI, y4 - yI, y4 - yII, y4 - yII;...
                    x1 - xI, x1 - xII, x1 - xI, x1 - xII;...
                    yI - y1, yI - y1, yII - y1, yII - y1;...
                    xI - x1, xII - x1, xI - x1, xII - x1;...
                    y1 - yI, y1 - yI, y1 - yII, y1 - yII;...
                    x2 - xI, x2 - xII, x2 - xI, x2 - xII];
                
                %Note:in Gauss quadrature, the differential transforms as dx = (l_x/2) d xi. Hence
                %we take the additional factor of sqrt(A)/2 onto B
                domainObj.d_N(:, :, e) = (1/(2*sqrt(domainObj.AEl(e))))*B';
            end
        end
        
        function domainObj = elementShapeFunctionArray(domainObj)
            %Gives values of element shape function arrays for Gauss quadrature
            %of convection matrix
            
            %Gauss points
            xi1 = -1/sqrt(3);
            xi2 = 1/sqrt(3);
            domainObj.NArray = zeros(4, 4, domainObj.nEl);
            for e = 1:domainObj.nEl
                %short hand notation
                x1 = domainObj.lc(e, 1, 1);
                x2 = domainObj.lc(e, 2, 1);
                y1 = domainObj.lc(e, 1, 2);
                y4 = domainObj.lc(e, 4, 2);
                
                %Coordinate transformation of Gauss quadrature points xi1 and xi2
                xI = 0.5*(x1 + x2) + 0.5*xi1*(x2 - x1);
                xII = 0.5*(x1 + x2) + 0.5*xi2*(x2 - x1);
                yI = 0.5*(y1 + y4) + 0.5*xi1*(y4 - y1);
                yII = 0.5*(y1 + y4) + 0.5*xi2*(y4 - y1);
                
                %Assuming bilinear shape functions here!!! See Fish&Belytschko p. 163
                N = [(xI - x2)*(yI - y4), (xII - x2)*(yI - y4), (xI - x2)*(yII - y4), (xII - x2)*(yII - y4);...
                    -(xI - x1)*(yI - y4), -(xII - x1)*(yI - y4), -(xI - x1)*(yII - y4), -(xII - x1)*(yII - y4);...
                    (xI - x1)*(yI - y1), (xII - x1)*(yI - y1), (xI - x1)*(yII - y1), (xII - x1)*(yII - y1);...
                    -(xI - x2)*(yI - y1), -(xII - x2)*(yI - y1), -(xI - x2)*(yII - y1), -(xII - x2)*(yII - y1)];
                
                %Note:in Gauss quadrature, the differential transforms as dx = (l_x/2) d xi. Hence
                %we take the additional factor of sqrt(A)/2 onto B
                domainObj.NArray(:, :, e) = (1/(2*sqrt(domainObj.AEl(e))))*N';
            end
        end

        function domainObj = setFluxForce(domainObj, qb)
            %Contribution to local force due to heat flux
            
            domainObj.fh = zeros(4, domainObj.nEl);
            
            for e = 1:domainObj.nEl
                xe(1) = domainObj.lc(e, 1, 1);
                xe(2) = domainObj.lc(e, 2, 1);
                xe(3) = domainObj.lc(e, 1, 2);
                xe(4) = domainObj.lc(e, 4, 2);
                N = @(x, y) domainObj.elementShapeFunctions(x, y, xe, domainObj.AEl(e));
                if(e <= domainObj.nElX && domainObj.naturalBoundaries(e, 1))
                    %lower boundary
                    q = @(x) qb{1}(x);
                    Nlo = @(x) N(x, 0);
                    fun = @(x) q(x)*Nlo(x);
                    domainObj.fh(:, e) = domainObj.fh(:, e) + integral(fun, xe(1), xe(2), 'ArrayValued', true);
                end
                if(mod(e, domainObj.nElX) == 0 && domainObj.naturalBoundaries(e, 2))
                    %right boundary
                    q = @(y) qb{2}(y);
                    Nr = @(y) N(1, y);
                    fun = @(y) q(y)*Nr(y);
                    domainObj.fh(:, e) = domainObj.fh(:, e) + integral(fun, xe(3), xe(4), 'ArrayValued', true);
                end
                if(e > (domainObj.nElY - 1)*domainObj.nElX && domainObj.naturalBoundaries(e, 3))
                    %upper boundary
                    q = @(x) qb{3}(x);
                    Nu = @(x) N(x, 1);
                    fun = @(x) q(x)*Nu(x);
                    domainObj.fh(:, e) = domainObj.fh(:, e) + integral(fun, xe(1), xe(2), 'ArrayValued', true);
                end
                if(mod(e, domainObj.nElX) == 1 && domainObj.naturalBoundaries(e, 4))
                    %left boundary
                    q = @(y) qb{4}(y);
                    Nle = @(y) N(0, y);
                    fun = @(y) q(y)*Nle(y);
                    domainObj.fh(:, e) = domainObj.fh(:, e) + integral(fun, xe(3), xe(4), 'ArrayValued', true);
                end
                
            end
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
            %this is wrong if lx, ly ~= 1 (size of domain)
            boundaryCoordinates = [[0 cumsum(domainObj.lElX(1:domainObj.nElX))], ones(1, domainObj.nElY - 1),...
                fliplr([0 cumsum(domainObj.lElX(1:domainObj.nElX))]), zeros(1, domainObj.nElY - 1);...
                zeros(1, domainObj.nElX + 1),...
                cumsum(domainObj.lElY(domainObj.nElX:domainObj.nElX:(domainObj.nElX*domainObj.nElY))),...
                ones(1, domainObj.nElX - 1),...
                fliplr(cumsum(domainObj.lElY(domainObj.nElX:domainObj.nElX:(domainObj.nElX*domainObj.nElY))))];
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
            domainObj = setFluxForce(domainObj, qb);
            domainObj = setNodalCoordinates(domainObj);
            domainObj = setBvec(domainObj);
            if domainObj.useConvection
                domainObj = domainObj.setConvectionMatrix;
            end
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
























