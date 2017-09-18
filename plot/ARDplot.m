%Plots "ARD" prior, gaussian likelihood and posterior in 2d

%Hyperparams
a = 300;
b = realmin;
prior_type = 'laplace';

%Likelihood params
mu = [-.6, -.9];
Sigma = 8e-4*eye(2);

%Plotting options
upperLim = 1;
lowerLim = -1;

meshPoints = 81;
[X, Y] = meshgrid(linspace(lowerLim, upperLim, meshPoints));
x = [X(:) Y(:)];

if strcmp(prior_type, 'ARD')
    l_prior = @(xx) a*log(b) - log(gamma(a)) - .5*log(2*pi) - (a + .5)*log(b + .5*xx(1)^2) + log(gamma(a + .5)) + ...
        a*log(b) - log(gamma(a)) - .5*log(2*pi) - (a + .5)*log(b + .5*xx(2)^2) + log(gamma(a + .5));
    l_prior = @(xx)  - (a + .5)*log(b + .5*xx(1)^2) - (a + .5)*log(b + .5*xx(2)^2);
elseif strcmp(prior_type, 'laplace')
    l_prior = @(xx) -a*sum(abs(xx));
end
l_likelihood = @(xx) -log(2*pi) - log(det(Sigma)) - .5*(xx - mu)*(Sigma\(xx - mu)');
l_likelihood = @(xx) - .5*(xx - mu)*(Sigma\(xx - mu)');

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
s1 = surf(X, Y, reshape(log_prior, meshPoints, meshPoints));
title('log prior')
s1.LineStyle = 'none';
axis tight
axis square
box on
subplot(1,3,2)
s2 = surf(X, Y, reshape(log_likelihood, meshPoints, meshPoints));
title('log likelihood')
s2.LineStyle = 'none';
axis tight
axis square
box on
subplot(1,3,3)
s3 = surf(X, Y, reshape(log_posterior, meshPoints, meshPoints));
title('log posterior')
s3.LineStyle = 'none';
axis tight
axis square
box on
