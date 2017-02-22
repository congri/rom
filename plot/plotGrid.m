function [] = plotGrid(Nc, Nf, lw)
%Plots the coarse grid onto microstructure plot

finePerCoarse = Nf/Nc;

if nargout < 3
    lw = 2;
end

hold on;
%x-direction
for x = (finePerCoarse + .5):finePerCoarse:(Nf - finePerCoarse + 1)
    line([x x], [0 Nf + .5], 'linewidth', lw, 'color', [1 1 1]);
end

%y-direction
for y = (finePerCoarse + .5):finePerCoarse:(Nf - finePerCoarse + 1)
    line([0 Nf + .5], [y y], 'linewidth', lw, 'color', [1 1 1]);
end

hold off;

end

