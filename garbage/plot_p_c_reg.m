%plot script to plot Xmean vs. p_c regression

f = figure;
k = 1;

if strcmp(romObj.mode, 'useLocal')
    for i = 1:romObj.coarseScaleDomain.nElX
        for j = 1:romObj.coarseScaleDomain.nElY
            subplot(romObj.coarseScaleDomain.nElX, romObj.coarseScaleDomain.nElY, k);
            for s = 1:romObj.nTrain
                plot(Phi.designMatrices{s}(k, 2*k), XMean(k, s), 'xb')
                hold on;
            end
            x = linspace(-7, 1, 100);
            y = romObj.theta_c.theta(2*k - 1) + romObj.theta_c.theta(2*k)*x;
            plot(x, y);
            axis tight;
            axis square;
            k = k + 1;
        end
    end
elseif strcmp(romObj.mode, 'none')
    for i = 1:romObj.coarseScaleDomain.nElX
        for j = 1:romObj.coarseScaleDomain.nElY
            for s = 1:romObj.nTrain
                plot(Phi.designMatrices{s}(k, 2), XMean(k, s), 'xb')
                hold on;
            end
            x = linspace(min(min(cell2mat(Phi.designMatrices))), max(max(cell2mat(Phi.designMatrices))), 100);
            y = romObj.theta_c.theta(1) + romObj.theta_c.theta(2)*x;
            plot(x, y);
            axis tight;
            k = k + 1;
        end
    end
end
