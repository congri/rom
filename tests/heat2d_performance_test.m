% performance test for heat2d
% clear;
addpath('./heatFEM');
addpath('./FEMgradient');

% set up domain object
nRes = 8;
gridVectorX = (1/nRes)*ones(1, nRes);
gridVectorY = (1/nRes)*ones(1, nRes);

domain = Domain(nRes, nRes, gridVectorX, gridVectorY);
domain.useConvection = false;

%Set up boundary condition functions
bc = [100 -200 150 -250];
boundaryTemperature = @(x) bc(1) + bc(2)*x(1) + bc(3)*x(2) + bc(4)*x(1)*x(2);
boundaryHeatFlux{1} = @(x) -(bc(3) + bc(4)*x);      %lower bound
boundaryHeatFlux{2} = @(y) (bc(2) + bc(4)*y);       %right bound
boundaryHeatFlux{3} = @(x) (bc(3) + bc(4)*x);       %upper bound
boundaryHeatFlux{4} = @(y) -(bc(2) + bc(4)*y);      %left bound

domain = domain.setBoundaries([2:(2*nRes + 2*nRes)], boundaryTemperature,...
    boundaryHeatFlux);

iterations = 10000;
% tic;
% for it = 1:iterations
%     out = heat2d(domain, D);
% end
% t = toc;
% 
% time_per_iteration = t/iterations


tic;
D = ones(nRes^2, 1);
for it = 1:iterations
    out = heat2d_v2(domain, D);
end
t = toc;

time_per_iteration_v2 = t/iterations

% analytical solution
[X, Y] = meshgrid(linspace(0, 1, nRes + 1));
T_true = 0*X;
for i = 1:numel(X)
    x = [X(i), Y(i)];
    T_true(i) = boundaryTemperature(x);
end

figure
subplot(1,2,1)
imagesc(out.Tff)
colorbar
subplot(1,2,2)
imagesc(T_true)
colorbar


