function [] = plotEffCond(nc, nf, meanEffCond, condTest)
%Plot mean effective conductivity and ground truth
figure
for i = 1:3
    %effective conductivities
    subplot(3,2,(2*i - 1))
    imagesc(reshape(meanEffCond(:, i), nc, nc));
    axis square
    grid off
    xticklabels({})
    yticklabels({})
    colorbar
    
    %ground truth
    subplot(3,2,2*i)
    imagesc(reshape(condTest(:, i), nf, nf));
    axis square
    grid off
    xticklabels({})
    yticklabels({})
    colorbar
    hold 
    plotGrid(nc, nf)
end
end

