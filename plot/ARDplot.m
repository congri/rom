%Plots "ARD" prior, gaussian likelihood and posterior in 2d

%Hyperparams
a = 1e-10;
b = 1e-10;

%Likelihood params
mu = [1, 1];
Sigma = 1e-3*eye(2);

%Plotting options
upperLim = 3;
lowerLim = -3;

meshPoints = 500;
[X, Y] = meshgrid(linspace(lowerLim, upperLim, meshPoints));
x = [X(:) Y(:)];

l_prior = @(xx) a*log(b) - log(gamma(a)) - .5*log(2*pi) - (a + .5)*log(b + .5*xx(1)^2) + log(gamma(a + .5)) + ...
    a*log(b) - log(gamma(a)) - .5*log(2*pi) - (a + .5)*log(b + .5*xx(2)^2) + log(gamma(a + .5));
l_likelihood = @(xx) -log(2*pi) - log(det(Sigma)) - .5*(xx - mu)*(Sigma\(xx - mu)');

N = size(x, 1);
log_prior = zeros(N, 1);
log_likelihood = log_prior;
log_posterior = log_prior;
for i = 1:N
    log_prior(i) = l_prior(x(i, :));
    log_likelihood(i) = l_likelihood(x(i, :));
    log_posterior(i) = log_prior(i) + log_likelihood(i);
end

f = figure;
subplot(1,3,1)
s1 = contourf(X, Y, reshape(log_prior, meshPoints, meshPoints));
axis tight
axis square
subplot(1,3,2)
s2 = contourf(X, Y, reshape(log_likelihood, meshPoints, meshPoints));
axis tight
axis square
subplot(1,3,3)
s3 = contourf(X, Y, reshape(log_posterior, meshPoints, meshPoints));
axis tight
axis square
