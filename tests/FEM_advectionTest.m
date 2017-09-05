%Advection-diffusion tests

clear all;
addpath('./heatFEM')

convergenceTest = true;
if(convergenceTest)
    ax = 20;
    conductivity = 1e1;
    Tbfun = @(x) (conductivity/ax)*exp((ax/conductivity)*x(1));
    %Gradient
    d_Tbfun = @(x) [exp((ax/conductivity)*x(1)); 0];
    qb{1} = @(x) 0;     %only for unit conductivity
    qb{2} = @(y) conductivity*exp(ax/conductivity);
    qb{3} = @(x) 0;
    qb{4} = @(y) -conductivity;
    
    %domain object. Best not change the order of commands!
    nX = 128;
    nY = 128;
    elXVec = (1/nX)*ones(1, nX);
    elYVec = (1/nY)*ones(1, nY);
    domain = Domain(nX, nY, elXVec, elYVec);
    domain.useConvection = true;
    domain = setBoundaries(domain, (2:(2*nX + 2*nY)), Tbfun, qb);
    convectionField = @(x) [ax; x(2)];
    convFieldArray = zeros(2, domain.nNodes);
    for n = 1:domain.nNodes
        convFieldArray(:, n) = convectionField(domain.nodalCoordinates(1:2, n));
    end
    
    
    
    %heat conductivity tensor for each element
    Dc = zeros(2, 2, domain.nEl);
    for j = 1:domain.nEl
        %Test is only valid for constant D in the whole domain!
        Dc(:, :, j) = conductivity*eye(2); %only isotropic material
    end

    out = heat2d(domain, Dc, convFieldArray);
    
    FEMtemperatureField = out.Tff;

    plt = true;
    if plt
        figure('units','normalized','outerposition',[0 0 1 1])
        subplot(1,2,1)
        imagesc(FEMtemperatureField);
        axis square
        grid off
        colorbar
        
        [X, Y] = meshgrid(linspace(0, 1, nX + 1));
        Ttrue = zeros(nX + 1);
        for i = 1:numel(X)
            Ttrue(i) = Tbfun([X(i) Y(i)]);
        end
        subplot(1,2,2)
        imagesc(Ttrue);
        axis square;
        grid off;
        colorbar
        
        diffSq = norm(Ttrue(:) - FEMtemperatureField(:))^2/numel(Ttrue)
    end
    
end