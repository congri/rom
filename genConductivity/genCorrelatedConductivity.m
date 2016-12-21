function [p, P] = genCorrelatedConductivity(domain, l1, l2, sigma_f2, nSamples)
%Generates spatially correlated random samples

%Compute coordinates of element centers
x = (domain.lElX/2):domain.lElX:(1 - (domain.lElX/2));
y = (domain.lElY/2):domain.lElY:(1 - (domain.lElY/2));
[X, Y] = meshgrid(x, y);
x = [X(:) Y(:)]';

log_sigma_f2 = log(sigma_f2);
log_l1 = log(l1);
log_l2 = log(l2);
params = [log_l1, log_l2, log_sigma_f2];

%squared distances in x and y direction
Xsq = sq_dist(x(1,:));
Ysq = sq_dist(x(2,:));

K = GPcov(Xsq, Ysq, 'ardSE', params);
m = zeros(1, domain.nEl);
p = mvnrnd(m, K, nSamples);
p = p';
P = reshape(p, domain.nElX, domain.nElY);
plt = false;
if plt
    figure
    subplot(1,3,1)
    contourf(X, Y, P)
    axis square
    subplot(1,3,2)
    pcolor(X,Y,P)
    axis square
    subplot(1,3,3)
    pcolor(X,Y, 1*(P > 0))
    axis square
end

end

