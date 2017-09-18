function [indexArrayOut] = padIndexMat(indexArray, padding)
%Pads logical array with further ones around areas with ones;
%used for padded macro-cells, i.e. also consider fine-scale pixels
%of the direct neighborhood of the macro-cell

[I, J] = find(indexArray);

%row lower bound
rlb = min(I) - padding;
if(rlb < 1)
    rlb = 1;
end
%row upper bound
rub = max(I) + padding;
if(rub > size(indexArray, 1))
    rub = size(indexArray, 1);
end
%column lower bound
clb = min(J) - padding;
if(clb < 1)
    clb = 1;
end
%column upper bound
cub = max(J) + padding;
if(cub > size(indexArray, 2))
    cub = size(indexArray, 2);
end

indexArrayOut = indexArray;
indexArrayOut(rlb:rub, clb:cub) = 1;


debug = false;
if debug
    subplot(1,2,1)
    imagesc(indexArray);
    xticks({});
    yticks({});
    axis square
    grid off
    title('Original array')
    
    subplot(1,2,2)
    imagesc(indexArrayOut);
    xticks({});
    yticks({});
    axis square
    grid off
    title('padded array')
    drawnow
    pause
end

end

