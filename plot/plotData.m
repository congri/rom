function [ output_args ] = plotData(cond, T, prnt)
%Plots finescale data samples

f = figure;
for i = 1:size(cond, 2)
    subplot(1,2,1)
    c = imagesc(reshape(cond(:, i), sqrt(size(cond, 1)), sqrt(size(cond, 1))));
    axis square
    grid off
    xticks({})
    yticks({})
    cbc = colorbar;
    cbc.Ticks = [min(min(cond)) max(max(cond))];
    cbc.TickLabelInterpreter = 'latex';
    tc = title('$\lambda_f$');
    tc.Interpreter = 'latex';
    
    subplot(1,2,2)
    t = imagesc(reshape(T(:, i), sqrt(size(T, 1)), sqrt(size(T, 1))));
    axis square
    grid off
    xticks({})
    yticks({})
    cbt = colorbar;
    cbt.TickLabelInterpreter = 'latex';
    tt = title('$U_f$')
    tt.Interpreter = 'latex';
    caxis([min(min(T)) max(max(T))])
    caxis([min(min(T)) 80])
    drawnow
    if prnt
        filename = strcat('~/images/siam17/dataPlot', num2str(i));
        print(f, filename, '-dpng', '-r300')
    end
    pause
end


end

