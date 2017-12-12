function [distanceMat] = computeDistanceMatrix(imageSize, point)
%Computes Gaussian distance to point

if nargin < 2
    point = .5*([imageSize(1), imageSize(2)] + 1);
end
[x, y] = meshgrid(1:imageSize(1), 1:imageSize(2));
xy = [x(:) y(:)];
distanceMat = -.5*sum((xy - point).^2, 2);
distanceMat = reshape(distanceMat, imageSize);

debug = false;
if debug
    figure
    imagesc((-distanceMat).^(1/64))
    title('A{^(1/64)}')
    drawnow
    pause
end


end

