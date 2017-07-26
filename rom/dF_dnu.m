function [F, dFdnu] = dF_dnu(nu, SigmaInv, XMean, theta, tau, designMatrices, nKernels)
%Gradient of lower bound w.r.t. kernel centers nu

% nu = reshape(nu, nKernels, size(designMatrices{1}, 2));

nCoarse = size(XMean, 1);
nTrain = numel(designMatrices);
kernelMatrix{1} = zeros(nCoarse, nKernels);  %prealloc
kernelMatrix = repmat(kernelMatrix, nTrain, 1);
kernelOffset = kernelMatrix;
dFdnu = 0*nu;
XModelData = 0*XMean;
for n = 1:nTrain
    for k = 1:nCoarse
        for m = 1:nKernels
            kernelOffset{n}(k, m) = norm(nu(m, :) - designMatrices{n}(k, :))^2;
            kernelMatrix{n}(k, m) = kernelFunction(tau(m), kernelOffset{n}(k, m), 'squaredExponential');
        end
    end
    XModelData(:, n) = kernelMatrix{n}*theta;
end



F = 0;
for n = 1:nTrain
    for k = 1:nCoarse
        F = F + .5*SigmaInv(k, k)*(XModelData(k,n)^2 - 2*XModelData(k, n)*XMean(k, n));
    end
end

FDcheck = false;
if(nargout > 1 || FDcheck)
    for m = 1:nKernels
        for n = 1:nTrain
            for k = 1:nCoarse
                dFdnu(m, :) = dFdnu(m, :) - 2*SigmaInv(k, k)*(XMean(k, n) - XModelData(k, n))*...
                    theta(m)*tau(m)*kernelMatrix{n}(k, m)*(designMatrices{n}(k, :) - nu(m, :)); %negative sign for fminunc
                %negative sign due to fminunc
            end
        end
    end
    
    %dFdnu
end

if FDcheck
    kernelMatrix{1} = zeros(nCoarse, nKernels);  %prealloc
    kernelMatrix = repmat(kernelMatrix, nTrain, 1);
    kernelOffset = kernelMatrix;
    XModelDataFD = 0*XMean;
    d = 1e-4;
    
    for i = 1:size(nu, 1)
        for j = size(nu, 2)
            nuFD = nu;
            nuFD(i, j) = nuFD(i, j) + d;
            for n = 1:nTrain
                for k = 1:nCoarse
                    for m = 1:nKernels
                        kernelOffset{n}(k, m) = norm(nuFD(m, :) - designMatrices{n}(k, :))^2;
                        kernelMatrix{n}(k, m) = kernelFunction(tau(m), kernelOffset{n}(k, m), 'squaredExponential');
                    end
                end
                XModelDataFD(:, n) = kernelMatrix{n}*theta;
            end
            
            
            FFD = 0;
            for n = 1:nTrain
                for k = 1:nCoarse
                    FFD = FFD + .5*SigmaInv(k, k)*(XModelDataFD(k,n)^2 - 2*XModelDataFD(k, n)*XMean(k, n));
                end
            end
            FDgrad(i, j) = (FFD - F)/d;
        end
    end    
        
    FDgrad
    grad = dFdnu
    relgrad = FDgrad./grad
    pause
end

end

