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
        f_tot                       %sum of fs and fh
        
        compute_grad = false        %should gradients be computed? memory
                                    %overhead!!
        d_loc_stiff                 %local stiffness gradient
        d_glob_stiff                %global stiffness gradient
        
        d_glob_force                %global force gradient
    end
    
    
    
    
    
    
    methods
        function self = Domain(nElX, nElY, lElX, lElY)
            %constructor
            %nElX andnElY are number of elements in x- and y-direction
            %lElX, lElY are vectors specifying element lengths in x- and y-directions. i-th element
            %in lElX: i-th column; j-th element in lElY: j-th row
            if nargin > 0
                self.nElX = nElX;
                if nargin > 1
                    self.nElY = nElY;
                end
            end
            %Set square mesh as default
            if nargin < 3
                lElX = self.lx/self.nElX*ones(1, self.nElX);
                lElY = self.ly/self.nElY*ones(1, self.nElY);
            end
            self.nEl = self.nElX*self.nElY;
            assert(numel(lElX) == self.nElX, 'incorrect number of elements specified in element length vector')
            assert(numel(lElY) == self.nElY, 'incorrect number of elements specified in element length vector')
            diffX = abs(sum(lElX) - self.lx);
            diffY = abs(sum(lElY) - self.ly);
            assert(diffX < eps, 'element lengths do not sum up to lx')
            assert(diffY < eps, 'element lengths do not sum up to ly')
            self.lElX = zeros(1, self.nEl);
            self.lElY = zeros(1, self.nEl);
            self.AEl = zeros(1, self.nEl);
            for e = 1:self.nEl
                self.lElX(e) = lElX(mod((e - 1), self.nElX) + 1);
                self.lElY(e) = lElY(floor((e - 1)/self.nElX) + 1);
                self.AEl(e) = self.lElX(e)*self.lElY(e);
            end
            self.cum_lElX = cumsum([0 lElX]);
            self.cum_lElY = cumsum([0 lElY]);
            self.nNodes = (self.nElX + 1)*(self.nElY + 1);
            self.boundaryNodes = int32([1:(self.nElX + 1),...
                2*(self.nElX + 1):(self.nElX + 1):(self.nElX + 1)*(self.nElY + 1),...
                ((self.nElX + 1)*(self.nElY + 1) - 1):(-1):((self.nElX + 1)*self.nElY + 1),...
                (self.nElX + 1)*((self.nElY - 1):(-1):1) + 1]);
            self.boundaryElements = int32([1:self.nElX,...
                2*(self.nElX):(self.nElX):(self.nElX*self.nElY),...
                ((self.nElX)*(self.nElY) - 1):(-1):(self.nElX*(self.nElY - 1) + 1),...
                (self.nElX)*((self.nElY - 2):(-1):1) + 1]);
            
            %local coordinate array. First index is element number, 2 is local node, 3 is x or y
            self = self.setLocCoord;
            self = self.setGlobalNodeNumber;
            
            self = setHeatSource(self, zeros(self.nEl, 1));  %zero as default
        end

        function self = setLocCoord(self)
            %Gives arrays taking the element and local node number and giving the nodal coordinate
            
            self.lc = zeros(self.nEl, 4, 2);
            for e = 1:self.nEl
                row = floor((e - 1)/self.nElX) + 1;
                col = mod((e - 1), self.nElX) + 1;

                %x-coordinates
                self.lc(e, 1, 1) = self.cum_lElX(col);
                self.lc(e, 2, 1) = self.cum_lElX(col + 1);
                self.lc(e, 3, 1) = self.lc(e, 2, 1);
                self.lc(e, 4, 1) = self.lc(e, 1, 1);
                
                %y-coordinates
                self.lc(e, 1, 2) = self.cum_lElY(row);
                self.lc(e, 2, 2) = self.lc(e, 1, 2);
                self.lc(e, 3, 2) = self.cum_lElY(row + 1);
                self.lc(e, 4, 2) = self.lc(e, 3, 2);
            end
        end

        function self = setGlobalNodeNumber(self)
            %Get global node number from global element number and local node number
            
            self.globalNodeNumber = zeros(self.nEl, 4, 'int32');
            for e = 1:self.nEl
                for l = 1:4
                    self.globalNodeNumber(e,1) = e + floor((e - 1)/self.nElX);
                    self.globalNodeNumber(e,2) = e + floor((e - 1)/self.nElX) + 1;
                    self.globalNodeNumber(e,3) = self.globalNodeNumber(e,1) + self.nElX + 2;
                    self.globalNodeNumber(e,4) = self.globalNodeNumber(e,1) + self.nElX + 1;
                end
            end
        end

        function self = setId(self)
            %put in equation number, get back global node number
            
            [eqs, i] = sort(self.nodalCoordinates(3, :));
            
            self.id = [eqs', i'];
            
            init = find(eqs == 1);
            
            self.id(1:(init-1), :) = [];
            
            self.id = self.id(:, 2);
            self.id = uint32(self.id);
        end

        function self = getEquations(self)
            %Equation number array for sparse global stiffness assembly
            
            localNodeInit = 1:4;
            %preallocate
            self.Equations = zeros(16*self.nEl, 2);
            self.LocalNode = zeros(16*self.nEl, 3);
            eq = 0; %equation number index
            for e = 1:self.nEl
                equationslm = self.lm(e, localNodeInit);
                equations = equationslm(equationslm > 0);
                localNode = localNodeInit(equationslm > 0);
                prevnEq = eq;
                eq = eq + numel(equations)^2;
                
                [Equations1, Equations2] = meshgrid(equations);
                self.Equations((prevnEq + 1):eq, :) = [Equations1(:) Equations2(:)];
                
                [LocalNode1, LocalNode2] = meshgrid(localNode);
                self.LocalNode((prevnEq + 1):eq, :) =...
                   [LocalNode1(:) LocalNode2(:) repmat(e, length(equations)^2, 1)];
            end
            
            %Shrink to fit
            self.Equations((eq + 1):end, :) = [];
            self.LocalNode((eq + 1):end, :) = [];
        end

        function self = getCoord(self)
            %Gives nodal coordinates in the first two rows and equation number from
            %global node number in the third row. Temperature of essential boundaries
            %is given in the fourth row, heat flux on natural boundaries in the fifth
            %Assign global node coordinates and equation numbers
            %In clockwise direction, the first node of every side is considered to belong to the boundary. The
            %last node is considered to belong to the next boundary. E.g. on a grid 5x5 nodes, nodes 1 - 4
            %belong to the lower boundary, nodes 5, 10, 15, 20 to the right, nodes 25, 24, 23, 22 to the upper
            %and 21, 16, 11, 6 to the left boundary.

            j = 1;  %equation number index
            self.nodalCoordinates = NaN*zeros(3, self.nNodes);
            for i = 1:self.nNodes
                row = floor((i - 1)/(self.nElX + 1)) + 1;
                col = mod((i - 1), (self.nElX + 1)) + 1;
                x = self.cum_lElX(col);
                y = self.cum_lElY(row);
                self.nodalCoordinates(1, i) = x;
                self.nodalCoordinates(2, i) = y;
                
                if(any(self.essentialNodes == i))
                    %essential node, no equation number assigned
                    self.nodalCoordinates(3, i) = 0;
                else
                    %Assign equation number j
                    self.nodalCoordinates(3, i) = j;
                    j = j + 1;
                end
            end 
        end

        function self = setNodalCoordinates(self)
            self = getCoord(self);
            self.lm = self.globalNodeNumber;
            for i = 1:size(self.globalNodeNumber, 1)
                for j = 1:size(self.globalNodeNumber, 2)
                    self.lm(i, j) = self.nodalCoordinates(3, self.globalNodeNumber(i, j));
                end
            end
            self = setId(self);
            self = getEquations(self);
            self.Equations = double(self.Equations);
            self.kIndex = sub2ind([4 4 self.nEl], self.LocalNode(:,1),...
                self.LocalNode(:,2), self.LocalNode(:,3));
        end

        function self = setBvec(self)
            self.nEq = max(self.nodalCoordinates(3,:));
            %Gauss points
            xi1 = -1/sqrt(3);
            xi2 = 1/sqrt(3);
            
            self.Bvec = zeros(8, 4, self.nEl);
            for e = 1:self.nEl
                for i = 1:4
                    self.essentialBoundary(i, e) =...
                        ~isnan(self.essentialTemperatures(self.globalNodeNumber(e, i)));
                end
                %short hand notation
                x1 = self.lc(e,1,1);
                x2 = self.lc(e,2,1);
                y1 = self.lc(e,1,2);
                y4 = self.lc(e,4,2);
                
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
                self.Bvec(:, :, e) = (1/(2*sqrt(self.AEl(e))))*[B1; B2; B3; B4];
            end
            
            self = self.get_loc_stiff_grad();
            if self.compute_grad
                %gradient precomputation
                self = self.get_glob_stiff_grad();
                self = self.get_glob_force_grad();
            end
        end
        
        function self = setConvectionMatrix(self)
            %Only call if necessary. This is memory consuming!
            disp('Setting convection matrix...')
            
            self = self.elementShapeFunctionArray;
            self = self.elementShapeFunctionGradients;
            self.convectionMatrix = zeros(4, 8, self.nEl);
            for e = 1:self.nEl
                self.convectionMatrix(:, :, e) = self.NArray(:, :, e)*self.d_N(:, :, e);
            end
            disp('done')
        end

        function self = setHeatSource(self, heatSourceField)
            %Gets the elements of the local force due to the heat source (an array with
            %input element number e and local node number i
            
            %Gauss points
            xi1 = -1/sqrt(3);
            xi2 = 1/sqrt(3);
            eta1 = -1/sqrt(3);
            eta2 = 1/sqrt(3);
            
            self.fs = zeros(4, self.nEl);

            for e = 1:self.nEl
                %short hand notation. Coordinates of local nodes
                x1 = self.lc(e, 1, 1);
                x2 = self.lc(e, 2, 1);
                y1 = self.lc(e, 1, 2);
                y4 = self.lc(e, 4, 2);
                
                %Coordinate transformation
                xI = 0.5*(x1 + x2) + 0.5*xi1*(x2 - x1);
                xII = 0.5*(x1 + x2) + 0.5*xi2*(x2 - x1);
                yI = 0.5*(y1 + y4) + 0.5*eta1*(y4 - y1);
                yII = 0.5*(y1 + y4) + 0.5*eta2*(y4 - y1);
                
                
                self.fs(1, e) = heatSourceField(e)*(1/self.AEl(e))*((xI - x2)*...
                    (yI - y4) + (xII - x2)*(yII - y4) + (xI - x2)*(yII - y4) + (xII - x2)*(yI - y4));
                self.fs(2, e) = -heatSourceField(e)*(1/self.AEl(e))*((xI - x1)*...
                    (yI - y4) + (xII - x1)*(yII - y4) + (xI - x1)*(yII - y4) + (xII - x1)*(yI - y4));
                self.fs(3, e) = heatSourceField(e)*(1/self.AEl(e))*((xI - x1)*...
                    (yI - y1) + (xII - x1)*(yII - y1) + (xI - x1)*(yII - y1) + (xII - x1)*(yI - y1));
                self.fs(4, e) = -heatSourceField(e)*(1/self.AEl(e))*((xI - x2)*...
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
        
        function self = elementShapeFunctionGradients(self)
            %Gives values of element shape function gradient arrays for Gauss quadrature
            %of convection matrix
            %This is similar to Bvec, but with different array arrangement
            
            %Gauss points
            xi1 = -1/sqrt(3);
            xi2 = 1/sqrt(3);
            self.d_N = zeros(4, 8, self.nEl);
            for e = 1:self.nEl
                %short hand notation
                x1 = self.lc(e, 1, 1);
                x2 = self.lc(e, 2, 1);
                y1 = self.lc(e, 1, 2);
                y4 = self.lc(e, 4, 2);
                
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
                self.d_N(:, :, e) = (1/(2*sqrt(self.AEl(e))))*B';
            end
        end
        
        function self = elementShapeFunctionArray(self)
            %Gives values of element shape function arrays for Gauss quadrature
            %of convection matrix
            
            %Gauss points
            xi1 = -1/sqrt(3);
            xi2 = 1/sqrt(3);
            self.NArray = zeros(4, 4, self.nEl);
            for e = 1:self.nEl
                %short hand notation
                x1 = self.lc(e, 1, 1);
                x2 = self.lc(e, 2, 1);
                y1 = self.lc(e, 1, 2);
                y4 = self.lc(e, 4, 2);
                
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
                self.NArray(:, :, e) = (1/(2*sqrt(self.AEl(e))))*N';
            end
        end

        function self = setFluxForce(self, qb)
            %Contribution to local force due to heat flux
            
            self.fh = zeros(4, self.nEl);
            
            for e = 1:self.nEl
                xe(1) = self.lc(e, 1, 1);
                xe(2) = self.lc(e, 2, 1);
                xe(3) = self.lc(e, 1, 2);
                xe(4) = self.lc(e, 4, 2);
                N = @(x, y) self.elementShapeFunctions(x, y, xe, self.AEl(e));
                if(e <= self.nElX && self.naturalBoundaries(e, 1))
                    %lower boundary
                    q = @(x) qb{1}(x);
                    Nlo = @(x) N(x, 0);
                    fun = @(x) q(x)*Nlo(x);
                    self.fh(:, e) = self.fh(:, e) + integral(fun, xe(1), xe(2), 'ArrayValued', true);
                end
                if(mod(e, self.nElX) == 0 && self.naturalBoundaries(e, 2))
                    %right boundary
                    q = @(y) qb{2}(y);
                    Nr = @(y) N(1, y);
                    fun = @(y) q(y)*Nr(y);
                    self.fh(:, e) = self.fh(:, e) + integral(fun, xe(3), xe(4), 'ArrayValued', true);
                end
                if(e > (self.nElY - 1)*self.nElX && self.naturalBoundaries(e, 3))
                    %upper boundary
                    q = @(x) qb{3}(x);
                    Nu = @(x) N(x, 1);
                    fun = @(x) q(x)*Nu(x);
                    self.fh(:, e) = self.fh(:, e) + integral(fun, xe(1), xe(2), 'ArrayValued', true);
                end
                if(mod(e, self.nElX) == 1 && self.naturalBoundaries(e, 4))
                    %left boundary
                    q = @(y) qb{4}(y);
                    Nle = @(y) N(0, y);
                    fun = @(y) q(y)*Nle(y);
                    self.fh(:, e) = self.fh(:, e) + integral(fun, xe(3), xe(4), 'ArrayValued', true);
                end
                
            end
        end

        function self = setBoundaries(self, natNodes, Tb, qb)    
            %natNodes holds natural nodes counted counterclockwise around domain, starting in lower
            %left corner. Tb and qb are function handles to temperature and heat flux boundary
            %functions
            self.boundaryType = true(1, 2*self.nElX + 2*self.nElY);
            self.boundaryType(natNodes) = false;
            self.essentialNodes = self.boundaryNodes(self.boundaryType);
            self.naturalNodes = int32(self.boundaryNodes(~self.boundaryType));
            
            %Set essential temperatures
            self.essentialTemperatures = NaN*ones(1, self.nNodes);
            %this is wrong if lx, ly ~= 1 (size of domain)
            boundaryCoordinates = [[0 cumsum(self.lElX(1:self.nElX))], ones(1, self.nElY - 1),...
                fliplr([0 cumsum(self.lElX(1:self.nElX))]), zeros(1, self.nElY - 1);...
                zeros(1, self.nElX + 1),...
                cumsum(self.lElY(self.nElX:self.nElX:(self.nElX*self.nElY))),...
                ones(1, self.nElX - 1),...
                fliplr(cumsum(self.lElY(self.nElX:self.nElX:(self.nElX*self.nElY))))];
            Tess = zeros(1, self.nNodes);
            for i = 1:(2*self.nElX + 2*self.nElY)
                Tess(i) = Tb(boundaryCoordinates(:, i));
            end
            self.essentialTemperatures(self.essentialNodes) = Tess(self.boundaryType);
            
            %Natural boundaries have to enclose natural nodes
            self.naturalBoundaries = false(self.nEl, 4);
            globNatNodes = self.boundaryNodes(natNodes);   %global node numbers of natural nodes
            
            %Set natural boundaries
            for i = 1:numel(globNatNodes)
                %find elements containing these nodes
                natElem = find(globNatNodes(i) == self.globalNodeNumber);
                [elem, ~] = ind2sub(size(self.globalNodeNumber), natElem);
                %find out side of boundary (lo, r, u, le)
                if(globNatNodes(i) == 1)
                    %lower left corner
                    assert(numel(elem) == 1, 'Error: corner node in more than one element?')
                    self.naturalBoundaries(1, 1) = true;
                    self.naturalBoundaries(1, 4) = true;
                elseif(globNatNodes(i) == self.nElX + 1)
                    %lower right corner
                    assert(numel(elem) == 1, 'Error: corner node in more than one element?')
                    self.naturalBoundaries(elem, 1) = true;
                    self.naturalBoundaries(elem, 2) = true;
                elseif(globNatNodes(i) == (self.nElX + 1)*(self.nElY + 1))
                    %upper right corner
                    assert(numel(elem) == 1, 'Error: corner node in more than one element?')
                    self.naturalBoundaries(elem, 2) = true;
                    self.naturalBoundaries(elem, 3) = true;
                elseif(globNatNodes(i) == (self.nElX + 1)*(self.nElY) + 1)
                    %upper left corner
                    assert(numel(elem) == 1, 'Error: corner node in more than one element?')
                    self.naturalBoundaries(elem, 3) = true;
                    self.naturalBoundaries(elem, 4) = true;
                elseif(globNatNodes(i) > 1 && globNatNodes(i) < self.nElX + 1)
                    %exclusively on lower bound
                    assert(numel(elem) == 2, 'Error: boundary node not in 2 elements?')
                    self.naturalBoundaries(elem(1), 1) = true;
                    self.naturalBoundaries(elem(2), 1) = true;
                elseif(mod(globNatNodes(i), self.nElX + 1) == 0)
                    %exclusively on right bound
                    assert(numel(elem) == 2, 'Error: boundary node not in 2 elements?')
                    self.naturalBoundaries(elem(1), 2) = true;
                    self.naturalBoundaries(elem(2), 2) = true;
                elseif(globNatNodes(i) > (self.nElX + 1)*(self.nElY) + 1)
                    %exclusively on upper bound
                    assert(numel(elem) == 2, 'Error: boundary node not in 2 elements?')
                    self.naturalBoundaries(elem(1), 3) = true;
                    self.naturalBoundaries(elem(2), 3) = true;
                elseif(mod(globNatNodes(i), self.nElX + 1) == 1)
                    %exclusively on left bound
                    assert(numel(elem) == 2, 'Error: boundary node not in 2 elements?')
                    self.naturalBoundaries(elem(1), 4) = true;
                    self.naturalBoundaries(elem(2), 4) = true;
                end
            end
            
            %Finally set local forces due to natural boundaries
            self = setFluxForce(self, qb);
            self.f_tot = self.fh + self.fs;
            self = setNodalCoordinates(self);
            self = setBvec(self);

        end

        function self = shrink(self)
            %To save memory. We use that on finescale domain to save memory
            self.lc = [];
            self.Equations = [];
            self.kIndex = [];
            self.boundaryNodes = [];
            self.essentialNodes = [];
            self.essentialTemperatures = [];
            self.naturalNodes = [];
            self.boundaryElements = [];
            self.naturalBoundaries = [];
            self.boundaryType = [];
            self.lx = [];
            self.ly = [];
            self.AEl = [];
            self.nEq = [];
            self.nodalCoordinates = [];
            self.globalNodeNumber =[];
            self.Bvec = [];
            self.essentialBoundary = [];
            self.lm = [];
            self.id = [];
            self.LocalNode = [];
            self.fs = [];
            self.fh = [];
        end
        
        function self = get_loc_stiff_grad(self)
            %Gives the local stiffness matrix gradient
            
            self.d_loc_stiff = zeros(4, 4, self.nEl);
            for e = 1:self.nEl
                self.d_loc_stiff(:, :, e) =...
                    self.Bvec(:, :, e)'*self.Bvec(:, :, e);
            end
        end
        
        function self = get_glob_stiff_grad(self)
            %gives global stiffness matrix gradient
            for e = 1:self.nEl
                grad_loc_k = zeros(4, 4, self.nEl);
                grad_loc_k(:, :, e) = self.d_loc_stiff(:, :, e);
                self.d_glob_stiff{e} = sparse(self.Equations(:, 1),...
                    self.Equations(:, 2), grad_loc_k(self.kIndex));
            end
        end
        
        function self = get_glob_force_grad(self)
            %compute global force gradient
            for e = 1:self.nEl
                self.d_glob_force{e} = get_glob_force_gradient(self, ...
                    self.d_loc_stiff(:, :, e), e);
            end
        end
    end
end
























