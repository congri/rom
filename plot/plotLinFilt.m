ind = reshape(1:16, 4, 4)';
ind = ind(:)';
for i = 1:16
    subplot(4,4,ind(i))
    p(i) = imagesc(reshape(w_all{i}, 64, 64));
    axis square
    grid off
    xticks({})
    yticks({})
    colorbar
    caxis([0 0.0246]);
    
%     %plot grid
%     for j = 1:3
%         line([0 257], [j*64 j*64], 'color', 'w', 'linewidth', 1);
%         line([j*64 j*64], [0 257], 'color', 'w', 'linewidth', 1);
%     end
end