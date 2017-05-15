%Test for 2d FEM code
clear all;
restoredefaultpath;
addpath('./params')
addpath('./heatFEM')
addpath('./plot')

patchTest = true;
if(patchTest)
    %Test temperature field given by function handle T. FEM solver should lead to the same solution.
    %ONLY MODIFY COEFFICIENTS a, DO NOT MODIFY FUNCTIONAL FORM OF T!!! Otherwise the test will fail
    a = [-4 3 2 6];
    Tbfun = @(x) a(1) + a(2)*x(1) + a(3)*x(2) + a(4)*x(1)*x(2);
    qb{1} = @(x) -(a(3) + a(4)*x);     %only for unit conductivity
    qb{2} = @(y) (a(2) + a(4)*y);
    qb{3} = @(x) (a(3) + a(4)*x);
    qb{4} = @(y) -(a(2) + a(4)*y);
    
    %domain object. Best not change the order of commands!
    nX = 5;
    nY = 5;
    domain = Domain(nX, nY, [.25 .25 .25 .125 .125], [.25 .25 .25 .125 .125]);
    domain = setBoundaries(domain, (2:(2*nX + 2*nY)), Tbfun, qb);
    
    %heat conductivity tensor for each element
    Dc = zeros(2, 2, domain.nEl);
    for j = 1:domain.nEl
        %Test is only valid for constant D in the whole domain!
        Dc(:,:,j) = eye(2); %only isotropic material
    end

    out = heat2d(domain, Dc);
    
    for i = 1:domain.nNodes
        Tcheck(mod(i - 1, nX + 1) + 1, floor((i - 1)/(nX + 1)) + 1) = Tbfun(domain.nodalCoordinates(1:2, i));
    end
    
    testTemperatureField = Tcheck';
    FEMtemperatureField = out.Tff;
    %for correct pcolor plot
    testTemperatureFieldPlot = zeros(size(testTemperatureField) + 1);
    testTemperatureFieldPlot(1:(end - 1), 1:(end - 1)) = testTemperatureField;
    FEMtemperatureFieldPlot = testTemperatureFieldPlot;
    FEMtemperatureFieldPlot(1:(end - 1), 1:(end - 1)) = FEMtemperatureField;
    diff = abs(testTemperatureField - FEMtemperatureField);
    diffPlot = testTemperatureFieldPlot;
    diffPlot(1:(end - 1), 1:(end - 1)) = diff;
    plt = true;
    if plt
        figure
        subplot(1, 3, 1)
        [Xgrid, Ygrid] = meshgrid(domain.cum_lElX, domain.cum_lElY);
        p1 = pcolor(Xgrid, Ygrid, testTemperatureField);
        p1.LineWidth = 2;
        p1.EdgeColor = [1 1 1];
        colorbar
        title('true temperature field')
        axis square
        subplot(1, 3, 2);
        p2 = pcolor(Xgrid, Ygrid, FEMtemperatureField);
        p2.LineWidth = 2;
        p2.EdgeColor = [1 1 1];
        colorbar
        title('FEM temperature field')
        axis square
        subplot(1, 3, 3);
        p3 = pcolor(Xgrid, Ygrid, diff);
        p3.LineWidth = 2;
        p3.EdgeColor = [1 1 1];
        colorbar
        title('difference')
        axis square
    end
    if(sqrt(sum(sum((testTemperatureField - FEMtemperatureField).^2)))/numel(testTemperatureField) > 1e-10)
        warning('Patch test for FEM failed')
        difference = sqrt(sum(sum((testTemperatureField - FEMtemperatureField).^2)))/numel(testTemperatureField)
    else
        difference = sqrt(sum(sum((testTemperatureField - FEMtemperatureField).^2)))/numel(testTemperatureField)
        disp('Patch test successful!')
    end
end




convergenceTest = false;
if(convergenceTest)
    % If no parallel pool exists, create one
    N_Threads = 16;
    if isempty(gcp('nocreate'))
        % Create with N_Threads workers
        parpool('local',N_Threads);
    end
    a = [-1 5 3 -4];
    c = 1; %c > 0
    d = 2;
    physicalc.Tbfun = @(x) d*log(norm(x + c)) + a(1) + a(2)*x(1)^2 + a(3)*x(2) + a(4)*x(1)*x(2);
    physicalc.gradT = @(x) [d*(x(1) + c)/norm(x + c)^2; d*(x(2) + c)/norm(x + c)^2]...
        + [2*a(2)*x(1) + a(4)*x(2); a(3) + a(4)*x(1)];
    
    nSimulations = 4;
    incrementFactor = 2;
    tic;
    for k = 1:nSimulations
        nX = nX*incrementFactor;
        nY = nY*incrementFactor;
        domainCell{k} = Domain(nX, nY);
        domainCell{k} = setBoundaries(domainCell{k}, (2:(2*nX + 2*nY)), Tbfun, qb);
        %specify boundary conditions here; only essential for this test
        l = 1/nX;
        boundaryCoordinates = [0:l:1, ones(1, nX), (1 - l):(-l):0, zeros(1, nX - 1);...
            zeros(1, nX + 1), l:l:1, ones(1, nX), (1 - l):(-l):l];
        %heat conductivity tensor for each element
        Dc = zeros(2, 2, domainCell{k}.nEl);
        for j = 1:domainCell{k}.nEl
            %Test is only valid for constant D in the whole domain!
            Dc(:,:,j) = eye(2); %only isotropic material
        end
%         for i = 1:4*nX
%             physical{k}.Tb(i) = physicalc.Tbfun(boundaryCoordinates(:, i));
%             qbtemp = - .25*Dc(:, :, 1)*physicalc.gradT(boundaryCoordinates(:, i));
%             %projection along normal vectors of domain boundaries
%             if i <= nX
%                 %bottom
%                 physical{k}.qb(i) = qbtemp(2);
%             elseif(mod(i, nX + 1) == 0 && i < (nX + 1)^2)
%                 %right
%                 physical{k}.qb(i) = -qbtemp(1);
%             elseif(i > nX*(nX + 1))
%                 %top
%                 physical{k}.qb(i) = -qbtemp(2);
%             elseif(mod(i, nX + 1) == 1 && i > 1)
%                 %left
%                 physical{k}.qb(i) = qbtemp(1);
%             end
%         end
%         physical{k}.boundaryType = true(1, 4*nX);         %true for essential node, false for natural node
%         physical{k}.essentialNodes = domainCell{k}.boundaryNodes(physical{k}.boundaryType);
%         physical{k}.naturalNodes = domainCell{k}.boundaryNodes(~physical{k}.boundaryType);
%         %Assign heat source field
%         physical{k}.heatSourceField = zeros(domainCell{k}.nEl, 1);
%         %Force contributions due to heat flux and source
%         physical{k}.fs = get_heat_source(physical{k}.heatSourceField, domainCell{k});
%         physical{k}.fh = get_flux_force(domainCell{k}, physical{k});
        for i = 1:domainCell{k}.nNodes
            TcheckConvergence{k}(mod(i - 1, nX + 1) + 1, floor((i - 1)/(nX + 1)) + 1) = physicalc.Tbfun(domainCell{k}.nodalCoordinates(1:2, i));
        end
        testTemperatureFieldConvergence{k} = TcheckConvergence{k}';
    end
    t1 = toc;
    parfor k = 1:nSimulations
    out = heat2d(domainCell{k}, Dc);
        FEMtemperatureFieldConvergence{k} = out.Tff;
        difference(k) = sqrt(sum(sum((testTemperatureFieldConvergence{k} -...
            FEMtemperatureFieldConvergence{k}).^2)))/numel(testTemperatureFieldConvergence{k});
        nElementsX(k) = domainCell{k}.nElX;
    end
    figure
    loglog(nElementsX, difference)
    xlabel('Number of elements')
    ylabel('Root mean square difference')
    title('Convergence to true solution')
    
    
end