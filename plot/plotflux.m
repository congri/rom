function [Qx, Qy] = plotflux(X, Y, Tff, domain, D)
%Plots the heat flux as quiver plot and gives the values back in matrices
%Qx, Qy

Qx = zeros(domain.nElX,domain.nElY);
Qy = zeros(domain.nElX,domain.nElY);
Qxt = zeros(domain.nElX,domain.nElY);
Qyt = zeros(domain.nElX,domain.nElY);
for i = 1:domain.nElX
   for j = 1:domain.nElY
      Qxt(i,j) = Tff(j,i+1) - Tff(j,i);
      Qyt(i,j) = Tff(j+1,i) - Tff(j,i);
      
      Qx(i,j) = -D(1,:,i + (j-1)*domain.nElX)*[Qxt(i,j); Qyt(i,j)];
      Qy(i,j) = -D(2,:,i + (j-1)*domain.nElX)*[Qxt(i,j); Qyt(i,j)]; 
   end
end

Qx = Qx'/domain.lElX;
Qy = Qy'/domain.lElY;

X(1,:) = [];
X(:,1) = [];
Y(1,:) = [];
Y(:,1) = [];

f = figure('name','Heat flux q');
set(f, 'Position', [735, 350, 720, 540]);
quiver(X,Y,Qx,Qy);
title('Heat flux q');
xlabel('x');
ylabel('y');
set(gca,'FontSize',14) 
xlim([0 1]);
ylim([0 1]);


end

