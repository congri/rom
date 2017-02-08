function [] = plotEffCond(nc, nf, meanEffCond, condTest)
%Plot mean effective conductivity and ground truth
figure
for i = 1:3
    
    %ground truth
    subplot(2,3, i)
    imagesc(reshape(condTest(:, i), nf, nf));
    axis square
    grid off
    xticklabels({})
    yticklabels({})
    xl = xlabel('x');
    yl = ylabel('y');
    xl.FontSize = 22;
    yl.FontSize = 22;
    if i == 2
        t1 = title('Finescale conductivity field');
        t1.FontSize = 28;
    end
    c1 = colorbar;
    c1.FontSize = 22;
    c1.Label.String = '\lambda';
    c1.Label.Rotation = 0;
    c1.Ticks = [min(min(condTest)) max(max(condTest))];
    hold;
    plotGrid(nc, nf)
    
    %effective conductivities
    subplot(2,3, i + 3)
    imagesc(reshape(meanEffCond(:, i), nc, nc));
%     caxis([min(min(meanEffCond)) max(max(meanEffCond))]);
    caxis([1 2]);
    axis square
    grid off
    xticklabels({})
    yticklabels({})
    xl = xlabel('x');
    yl = ylabel('y');
    xl.FontSize = 22;
    yl.FontSize = 22;
    if i == 2
        t2 = title('Coarse-grained conductivity field');
        t2.FontSize = 28;
    end
    c2 = colorbar;
    c2.FontSize = 22;
    c2.Label.String = '\Lambda';
    c2.Label.Rotation = 0;
    c2.Ticks = [1 1.2 1.4 1.6 1.8 2];
    
end
end

