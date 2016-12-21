function [fh] = get_flux_force(domain, qb)
%Contribution to local force due to heat flux

fh = zeros(4, domain.nEl);

for e = 1:domain.nEl
    xe(1) = domain.lc(e, 1, 1);
    xe(2) = domain.lc(e, 2, 1);
    xe(3) = domain.lc(e, 1, 2);
    xe(4) = domain.lc(e, 4, 2);
    N = @(x,y) elementShapeFunctions(x, y, xe, domain.AEl);
    if(e <= domain.nElX && domain.naturalBoundaries(e, 1))
        %lower boundary
        q = @(x) qb{1}(x);
        Nlo = @(x) N(x, 0);
        fun = @(x) q(x)*Nlo(x);
        fh(:, e) = fh(:, e) + integral(fun, xe(1), xe(2), 'ArrayValued', true);
    end
    if(mod(e, domain.nElX) == 0 && domain.naturalBoundaries(e, 2))
        %right boundary
        q = @(y) qb{2}(y);
        Nr = @(y) N(1, y);
        fun = @(y) q(y)*Nr(y);
        fh(:, e) = fh(:, e) + integral(fun, xe(3), xe(4), 'ArrayValued', true);
    end
    if(e > (domain.nElY - 1)*domain.nElX && domain.naturalBoundaries(e, 3))
        %upper boundary
        q = @(x) qb{3}(x);
        Nu = @(x) N(x, 1);
        fun = @(x) q(x)*Nu(x);
        fh(:, e) = fh(:, e) + integral(fun, xe(1), xe(2), 'ArrayValued', true);
    end
    if(mod(e, domain.nElX) == 1 && domain.naturalBoundaries(e, 4))
        %left boundary
        q = @(y) qb{4}(y);
        Nle = @(y) N(0, y);
        fun = @(y) q(y)*Nle(y);
        fh(:, e) = fh(:, e) + integral(fun, xe(3), xe(4), 'ArrayValued', true);
    end
    
end



%OLD VERSIONS
% %Gauss points
% xi1 = -1/sqrt(3);
% xi2 = 1/sqrt(3);
% eta1 = -1/sqrt(3);
% eta2 = 1/sqrt(3);
% 
% lElY = domain.lElY;
% 
% for e = 1:domain.nEl
%     %Contribution due to free boundaries
%     
%     %short hand notation. Coordinates of local nodes
%     x1 = domain.lc(e,1,1);
%     x2 = domain.lc(e,2,1);
%     y1 = domain.lc(e,1,2);
%     y4 = domain.lc(e,4,2);
%     
%     %Coordinate transformation
%     xI = 0.5*(x1 + x2) + 0.5*xi1*(x2 - x1);
%     xII = 0.5*(x1 + x2) + 0.5*xi2*(x2 - x1);
%     yI = 0.5*(y1 + y4) + 0.5*eta1*(y4 - y1);
%     yII = 0.5*(y1 + y4) + 0.5*eta2*(y4 - y1);
%     
%     %Check if element belongs to natural boundary
%     nat = false;
%     for i = 1:4
%         if(~isnan(domain.nodalCoordinates(5,domain.globalNodeNumber(e,i)))) %if it is a number, a heat flux is assigned --> natural boundary
%             nat = true;
%             break;
%         end
%     end
% 
%     
%     if nat      %The element has natural boundaries
% %         if(e == 1) %lower left corner
% %             if(strcmp(domain.boundaries(1),'natural'))
% %                 h = physical.qb(1);
% %                 fh(1,e) = fh(1,e) + h*(1/(2*lElY))*(-y4)*(xI + xII - 2*x2);
% %                 fh(2,e) = fh(2,e) + h*(1/(2*lElY))*y4*(xI + xII - 2*x1);
% %             elseif(strcmp(domain.boundaries(4), 'natural'))
% %                 h = physical.qb(4);
% %                 fh(1,e) = fh(1,e) - h*(1/2)*(yI + yII - 2*y4);
% %                 fh(4,e) = fh(4,e) + h*(1/2)*(yI + yII - 2*y1);
% %             else
% %                 error('Which boundary of corner element is natural?');
% %             end
% %             
% %         elseif(e == domain.nElX) %lower right corner
% %             if(strcmp(domain.boundaries(1),'natural'))
% %                 h = physical.qb(1);
% %                 fh(1,e) = fh(1,e) + h*(1/(2*lElY))*(-y4)*(xI + xII - 2*x2);
% %                 fh(2,e) = fh(2,e) + h*(1/(2*lElY))*y4*(xI + xII - 2*x1);
% %             elseif(strcmp(domain.boundaries(2), 'natural'))
% %                 h = physical.qb(2);
% %                 fh(2,e) = fh(2,e) - h*(1/2)*(yI + yII - 2*y4);
% %                 fh(3,e) = fh(3,e) + h*(1/2)*(yI + yII - 2*y1);
% %             else
% %                 error('Which boundary of corner element is natural?');
% %             end
% %             
% %         elseif(e == domain.nElX*domain.nElY) %upper right corner
% %             if(strcmp(domain.boundaries(3),'natural'))
% %                 h = physical.qb(3);
% %                 fh(3,e) = fh(3,e) + h*(1/2)*(xI + xII - 2*x1);
% %                 fh(4,e) = fh(4,e) - h*(1/2)*(xI + xII - 2*x2);
% %             elseif(strcmp(domain.boundaries(2), 'natural'))
% %                 h = physical.qb(2);
% %                 fh(2,e) = fh(2,e) - h*(1/2)*(yI + yII - 2*y4);
% %                 fh(3,e) = fh(3,e) + h*(1/2)*(yI + yII - 2*y1);
% %             else
% %                 error('Which boundary of corner element is natural?');
% %             end
% %             
% %         elseif(e == domain.nElX*(domain.nElY - 1) + 1) %upper left corner
% %             if(strcmp(domain.boundaries(3),'natural'))
% %                 h = physical.qb(3);
% %                 fh(3,e) = fh(3,e) + h*(1/2)*(xI + xII - 2*x1);
% %                 fh(4,e) = fh(4,e) - h*(1/2)*(xI + xII - 2*x2);
% %             elseif(strcmp(domain.boundaries(4), 'natural'))
% %                 h = physical.qb(4);
% %                 fh(1,e) = fh(1,e) - h*(1/2)*(yI + yII - 2*y4);
% %                 fh(4,e) = fh(4,e) + h*(1/2)*(yI + yII - 2*y1);
% %             else
% %                 error('Which boundary of corner element is natural?');
% %             end
% %             
% %         elseif(e > 1 && e < domain.nElX && strcmp(domain.boundaries(1),'natural')) %element is on lower boundary
% %             h = physical.qb(1);
% %             %Second order Gauss quadrature for linear function?
% %             fh(1,e) = fh(1,e) + h*(1/(2*lElY))*(-y4)*(xI + xII - 2*x2);
% %             fh(2,e) = fh(2,e) + h*(1/(2*lElY))*y4*(xI + xII - 2*x1);
% %         elseif(mod(e, domain.nElX) == 0 && e ~= domain.nElX && e ~= domain.nEl && strcmp(domain.boundaries(2), 'natural')) %element is on right boundary
% %             h = physical.qb(2);
% %             fh(2,e) = fh(2,e) - h*(1/2)*(yI + yII - 2*y4);
% %             fh(3,e) = fh(3,e) + h*(1/2)*(yI + yII - 2*y1);
% %         elseif(e > domain.nElX*(domain.nElY - 1) + 1 && e < domain.nEl && strcmp(domain.boundaries(3),'natural')) %element is on upper boundary
% %             h = physical.qb(3);
% %             fh(3,e) = fh(3,e) + h*(1/2)*(xI + xII - 2*x1);
% %             fh(4,e) = fh(4,e) - h*(1/2)*(xI + xII - 2*x2);
% %         elseif(mod(e, domain.nElX) == 1 && e ~= 1 && e ~= domain.nElX*(domain.nElY - 1) + 1 && strcmp(domain.boundaries(4),'natural')) %element is on left boundary
% %             h = physical.qb(4);
% %             fh(1,e) = fh(1,e) - h*(1/2)*(yI + yII - 2*y4);
% %             fh(4,e) = fh(4,e) + h*(1/2)*(yI + yII - 2*y1);
% %         else
% %             error('Between which nodes is this boundary?');
% %         end
%         
%         for l = 1:4     %loop through local node numbers of element e
%             globalNode = domain.globalNodeNumber(e, l);
%             if(any(globalNode == physical.naturalNodes))
%                 %node is natural
%                 if globalNode < domain.nElX + 1
%                     %lower boundary
%                     h = physical.qb(globalNode == domain.boundaryNodes);
%                     if l == 1
%                         fh(1,e) = fh(1,e) + h*(1/(2*lElY))*(-y4)*(xI + xII - 2*x2);
%                     elseif l == 2
%                         fh(2,e) = fh(2,e) + h*(1/(2*lElY))*y4*(xI + xII - 2*x1);
%                     else
%                         error('Impossible combination of boundary and local node number')
%                     end
%                 elseif(mod(globalNode, domain.nElX + 1) == 0 && globalNode ~= domain.nNodes)
%                     %right boundary
%                     h = physical.qb(globalNode == domain.boundaryNodes);
%                     if l == 2
%                         fh(2, e) = fh(2,e) - h*(1/2)*(yI + yII - 2*y4);
%                     elseif l == 3
%                         fh(3, e) = fh(3,e) + h*(1/2)*(yI + yII - 2*y1);
%                     else
%                         error('Impossible combination of boundary and local node number')
%                     end
%                 elseif(globalNode > domain.nElY*(domain.nElX + 1) + 1)
%                     %upper boundary
%                     h = physical.qb(globalNode == domain.boundaryNodes);
%                     if l == 3
%                         fh(3,e) = fh(3,e) + h*(1/2)*(xI + xII - 2*x1);
%                     elseif l == 4
%                         fh(4,e) = fh(4,e) - h*(1/2)*(xI + xII - 2*x2);
%                     else
%                         error('Impossible combination of boundary and local node number')
%                     end
%                 elseif(mod(globalNode, domain.nElX + 1) == 1 && globalNode ~= 1)
%                     %left boundary
%                     h = physical.qb(globalNode == domain.boundaryNodes);
%                     if l == 1
%                         fh(1,e) = fh(1,e) - h*(1/2)*(yI + yII - 2*y4);
%                     elseif l == 4
%                         fh(4,e) = fh(4,e) + h*(1/2)*(yI + yII - 2*y1);
%                     else
%                         error('Impossible combination of boundary and local node number')
%                     end
%                 end
%             end
%             
%         end
%     end
% end

end

