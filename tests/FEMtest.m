%Test for 2d FEM code
clear all;
addpath('~/matlab/projects/cgrom2d/params')
addpath('~/matlab/projects/cgrom2d/heatFEM')
addpath('~/matlab/projects/cgrom2d/plot')

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
    nc = 64;
    domainc = Domain(nc, nc, 1, 1);
    domainc = setBoundaries(domainc, (2:(4*nc)), Tbfun, qb);
%     domainc = setNodalCoordinates(domainc);
%     domainc = setBvec(domainc);
%     domainc = setHeatSource(domainc, zeros(domainc.nEl, 1));
    
    %heat conductivity tensor for each element
    Dc = zeros(2, 2, domainc.nEl);
    for j = 1:domainc.nEl
        %Test is only valid for constant D in the whole domain!
        Dc(:,:,j) = eye(2); %only isotropic material
    end

    out = heat2d(domainc, Dc);
    
    for i = 1:domainc.nNodes
        Tcheck(mod(i - 1, nc + 1) + 1, floor((i - 1)/(nc + 1)) + 1) = Tbfun(domainc.nodalCoordinates(1:2, i));
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
    plt = false;
    if plt
        figure
        subplot(1, 3, 1)
        pcolor(testTemperatureFieldPlot);
        colorbar
        title('true temperature field')
        axis square
        subplot(1, 3, 2);
        pcolor(FEMtemperatureFieldPlot);
        colorbar
        title('FEM temperature field')
        axis square
        subplot(1, 3, 3);
        pcolor(diffPlot)
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
    
    nSimulations = 22;
    incrementFactor = 4;
    tic;
    for k = 1:nSimulations
        nc = incrementFactor*k;
        domain{k} = Domain(nc, nc, 1, 1);
        %specify boundary conditions here; only essential for this test
        l = 1/nc;
        boundaryCoordinates = [0:l:1, ones(1, nc), (1 - l):(-l):0, zeros(1, nc - 1);...
            zeros(1, nc + 1), l:l:1, ones(1, nc), (1 - l):(-l):l];
        %heat conductivity tensor for each element
        Dc = zeros(2, 2, domain{k}.nEl);
        for j = 1:domain{k}.nEl
            %Test is only valid for constant D in the whole domain!
            Dc(:,:,j) = eye(2); %only isotropic material
        end
        for i = 1:4*nc
            physical{k}.Tb(i) = physicalc.Tbfun(boundaryCoordinates(:, i));
            qbtemp = - .25*Dc(:, :, 1)*physicalc.gradT(boundaryCoordinates(:, i));
            %projection along normal vectors of domain boundaries
            if i <= nc
                %bottom
                physical{k}.qb(i) = qbtemp(2);
            elseif(mod(i, nc + 1) == 0 && i < (nc + 1)^2)
                %right
                physical{k}.qb(i) = -qbtemp(1);
            elseif(i > nc*(nc + 1))
                %top
                physical{k}.qb(i) = -qbtemp(2);
            elseif(mod(i, nc + 1) == 1 && i > 1)
                %left
                physical{k}.qb(i) = qbtemp(1);
            end
        end
        physical{k}.boundaryType = true(1, 4*nc);         %true for essential node, false for natural node
%         physical{k}.boundaryType([(2:nc), (nc + 2):(2*nc)]) = false;           %lower boundary is natural
        physical{k}.essentialNodes = domain{k}.boundaryNodes(physical{k}.boundaryType);
        physical{k}.naturalNodes = domain{k}.boundaryNodes(~physical{k}.boundaryType);
        domain{k} = setNodalCoordinates(domain{k}, physical{k});
        domain{k} = setBvec(domain{k});
        %Assign heat source field
        physical{k}.heatSourceField = zeros(domain{k}.nEl, 1);
        %Force contributions due to heat flux and source
        physical{k}.fs = get_heat_source(physical{k}.heatSourceField, domain{k});
        physical{k}.fh = get_flux_force(domain{k}, physical{k});
        for i = 1:domain{k}.nNodes
            Tcheck{k}(mod(i - 1, nc + 1) + 1, floor((i - 1)/(nc + 1)) + 1) = physicalc.Tbfun(domain{k}.nodalCoordinates(1:2, i));
        end
        testTemperatureField{k} = Tcheck{k}';
    end
    t1 = toc;
    parfor k = 1:nSimulations
    out = heat2d(domain{k}, physical{k}, Dc);
        FEMtemperatureField{k} = out.Tff;
        difference(k) = sqrt(sum(sum((testTemperatureField{k} - FEMtemperatureField{k}).^2)))/numel(testTemperatureField{k});
        nElementsX(k) = domain{k}.nElX;
    end
    figure
    loglog(nElementsX, difference)
    xlabel('Number of elements')
    ylabel('Root mean square difference')
    title('Convergence to true solution')
    
    
end