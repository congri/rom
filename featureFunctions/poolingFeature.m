function [pooledLambda] = poolingFeature(lambda, element, wndw, poolfunc, pooledPixel, stride, padding)
%Feature function that takes lambda as an argument gives back pooled elements (i.e. pixels of the
%pooled image)

%only works for square elements!!
lambda = reshape(lambda, sqrt(length(lambda)), sqrt(length(lambda)));

%pooling
if nargin < 7
    [pooledLambda] = pool2d(lambda, wndw, poolfunc, pooledPixel, stride);
else
    [pooledLambda] = pool2d(lambda, wndw, poolfunc, pooledPixel, stride, padding);
end

end

