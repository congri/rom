function [pooledImage] = pool2d(pic, wndw, poolfunc, pooledPixel, stride, padding)
%Pooling (downsampling) of 2d image
%   pic:        original 2d image
%   wndw:       sliding window
%   poolfunc:   pooling function, window picture --> scalar
%   stride:     window step width
%   padding:    inserting zeros around the picture
%   pooledPixel:     if positive, only compute value of pooledPixel pixel

if nargin < 5
    %Default stride/ padding
    stride = [1 1];
    padding = 0;
elseif nargin < 6
    padding = 0;
end

if pooledPixel < 1
    pooledImage = zeros(numel(1:stride(1):size(pic, 1)), numel(1:stride(2):size(pic, 2)));
end
row_pool = 1;
column_pool = 1;
pPixel = 1;
for row = 1:stride(1):(size(pic, 1) - wndw(1) + 1)
    for column = 1:stride(2):(size(pic, 2) - wndw(2) + 1)
        if pooledPixel < 1
            %row and column give the coordinate of the upper left corner of the filter window
            window_pic = pic(row:(row + wndw(1) - 1), column:(column + wndw(2) - 1));
            pooledImage(row_pool, column_pool) = poolfunc(window_pic);
            column_pool = column_pool + 1;
        elseif(pooledPixel == pPixel)
            %row and column give the coordinate of the upper left corner of the filter window
            window_pic = pic(row:(row + wndw(1) - 1), column:(column + wndw(2) - 1));
            pooledImage = poolfunc(window_pic);
        else
            %just pass
        end
        pPixel = pPixel + 1;
    end
    column_pool = 1;
    row_pool = row_pool + 1;
end


end

