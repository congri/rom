%Advection-diffusion tests

clear all;
addpath('./heatFEM')

convergenceTest = true;
if(convergenceTest)
    %% Set random parameters
    discriminant = sqrt(-1);
    while(~isreal(discriminant))
        %constant convection field
        ax = normrnd(0, 10)
        ay = normrnd(0, 10)
        %constant conductivity field
        conductivity = lognrnd(1, 1)
        %From T = e^(kx*x + ky*y), fix kx
        kx = normrnd(0, 1)
        discriminant = sqrt(ay^2 - 4*conductivity*(conductivity*kx^2 - ax*kx))
    end
    %Both + or - are possible
%     ky = (1/(2*conductivity))*(ay + discriminant);
    ky = (1/(2*conductivity))*(ay - discriminant)
    
    %% Set handles to true temperature field + gradient
    Tbfun = @(x) exp(kx*x(1) + ky*x(2));
    %Gradient
    d_Tbfun = @(x) [kx*Tbfun(x); ky*Tbfun(x)];
    qb{1} = @(x) -conductivity*[0 1]*d_Tbfun([x; 0]);
    qb{2} = @(y) conductivity*[1 0]*d_Tbfun([1; y]);
    qb{3} = @(x) conductivity*[0 1]*d_Tbfun([x; 1]);
    qb{4} = @(y) -conductivity*[1 0]*d_Tbfun([0; y]);
    
    %loop over different system sizes to check convergence
    iter = 1;
    plt = true;
    if plt
        figure('units','normalized','outerposition',[0 0 1 1])
    end
    nStart = 1;
    nEnd = 9;
    nIter = nEnd - nStart + 1;
    for nn = nStart:nEnd
        %% Set up domain object + convection field. Best not change the order of commands!
        nX = 2^nn;
        nY = nX;
        elXVec = (1/nX)*ones(1, nX);
        elYVec = (1/nY)*ones(1, nY);
        domain = Domain(nX, nY, elXVec, elYVec);
        domain.useConvection = true;
        domain = setBoundaries(domain, (2:(2*nX + 2*nY)), Tbfun, qb);
        convectionField = @(x) [ax; ay];
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
        
        %% Solve FEM
        out = heat2d(domain, Dc, convFieldArray);
        
        [X, Y] = meshgrid(linspace(0, 1, nX + 1));
        Ttrue = zeros(nX + 1);
        for i = 1:numel(X)
            Ttrue(i) = Tbfun([X(i) Y(i)]);
        end
        FEMtemperatureField = out.Tff;
        
        diffSq(iter) = norm(Ttrue(:) - FEMtemperatureField(:))^2/(norm(Ttrue(:))^2)
        %% Plot true and FEM
        if plt
            subplot(nIter, 2, 2*iter - 1)
            s1 = imagesc(FEMtemperatureField);
%             s1.LineStyle = 'none';
            axis square
            axis tight
            xticks({});
            yticks({});
%             colorbar
            
            subplot(nIter,2,2*iter)
            s2 = imagesc(Ttrue);
%             s2.LineStyle = 'none';
            axis square;
            axis tight;
            xticks({});
            yticks({});
%             colorbar
        end
        iter = iter + 1;
    end
end