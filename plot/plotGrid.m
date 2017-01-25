function [] = plotGrid(Nc, Nf)
%Plots the coarse grid onto microstructure plot

finePerCoarse = Nf/Nc;

hold on;
%x-direction
for x = (finePerCoarse + .5):finePerCoarse:(Nf - finePerCoarse + 1)
    line([x x], [0 Nf], 'linewidth', 2, 'color', [1 1 1]);
end

%y-direction
for y = (finePerCoarse + .5):finePerCoarse:(Nf - finePerCoarse + 1)
    line([0 Nf], [y y], 'linewidth', 2, 'color', [1 1 1]);
end

hold off;

end

