function [X] = gaussLinFilt(lambda, muGaussFilt, sigmaGaussFilt)
%Gaussian shaped linear filter centered at the element center

if nargin < 2
    muGaussFilt = [(size(lambda, 1) + 1)/2 (size(lambda, 2) + 1)/2];
end
if nargin < 3
    sigmaGaussFilt = 10*[size(lambda, 1) size(lambda, 2)];
end
[x, y] = meshgrid(1:size(lambda, 1), 1:size(lambda, 2));
xy = [x(:) y(:)];
w = mvnpdf(xy, muGaussFilt, sigmaGaussFilt);
w = w/norm(w, 1);
w = reshape(w, size(lambda, 1), size(lambda, 2));

debug = false;
if debug
    figure
    imagesc(w)
    drawnow
    pause
end

X = sum(sum(w.*lambda));


end

