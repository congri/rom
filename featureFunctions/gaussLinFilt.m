function [X] = gaussLinFilt(lambda, muGaussFilt, sigmaGaussFiltFactor)
%Gaussian shaped linear filter centered at the element center

if(nargin < 2 || any(isnan(muGaussFilt)))
    muGaussFilt = [(size(lambda, 1) + 1)/2 (size(lambda, 2) + 1)/2];
end
if nargin < 3
    sigmaGaussFiltFactor = 10;
end
sigmaGaussFilt = sigmaGaussFiltFactor*[size(lambda, 1) size(lambda, 2)].^2;

[x, y] = meshgrid(1:size(lambda, 1), 1:size(lambda, 2));
xy = [x(:) y(:)];
w = mvnpdf(xy, muGaussFilt, sigmaGaussFilt);
w = w/sum(w);
w = reshape(w, size(lambda, 1), size(lambda, 2));

X = sum(sum(w.*lambda));

debug = false;
if debug
    figure
    subplot(1,3,1)
    title('w')
    imagesc(w)
    axis square
    grid off
    
    subplot(1,3,2)
    title('lambda')
    imagesc(lambda)
    axis square
    grid off
    
    subplot(1,3,3)
    title('w.*lambda')
    imagesc(w.*lambda)
    axis square
    grid off
    drawnow
    X
    pause
end


end

