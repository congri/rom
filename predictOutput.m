function [Tf_mean, TfVarArray, Tf_mean_tot, Tf_sq_mean_tot, meanMahaErr, meanSqDist, sqDist, meanEffCond, meanSqDistErr] =...
    predictOutput(nSamples_p_c, testSample_lo, testSample_up, testFilePath, modelParamsFolder, mode)
%Function to predict finescale output from generative model

%Load test file
Tffile = matfile(testFilePath);
if nargout > 4
    Tf = Tffile.Tf(:, testSample_lo:testSample_up);
end
[theta_c, theta_cf, domainc, domainf, phi, featureFunctionMean, featureFunctionSqMean,...
    featureFunctionMin, featureFunctionMax] = loadTrainedParams(modelParamsFolder);

if strcmp(mode, 'linFiltSeq')
    w = dlmread('./data/w');
    for i = 1:size(w, 1)
        phi{end + 1} = @(lambda) sum(w(i, :)'.*log(lambda));
    end
end

addpath('./rom')
addpath('./aux')

%% Compute design matrices
Phi = DesignMatrix(domainf, domainc, phi, Tffile, testSample_lo:testSample_up);
load('./data/conductivityTransformation.mat');
%change here for anisotropy!
condTransOpts.anisotropy = false;
Phi = Phi.computeDesignMatrix(domainc.nEl, domainf.nEl, condTransOpts, mode);
%Normalize design matrices
%Phi = Phi.standardizeDesignMatrix(featureFunctionMean, featureFunctionSqMean);
% Phi = Phi.rescaleDesignMatrix(featureFunctionMin, featureFunctionMax);
if strcmp(mode, 'useNeighbor')
    %use feature function information from nearest neighbors
    Phi = Phi.includeNearestNeighborFeatures([domainc.nElX domainc.nElY]);
elseif strcmp(mode, 'useLocalNeighbor')
    Phi = Phi.includeLocalNearestNeighborFeatures([domainc.nElX domainc.nElY]);
elseif strcmp(mode, 'useLocalDiagNeighbor')
    Phi = Phi.includeLocalDiagNeighborFeatures([domainc.nElX domainc.nElY]);
elseif strcmp(mode, 'useDiagNeighbor')
    %use feature function information from nearest and diagonal neighbors
    Phi = Phi.includeDiagNeighborFeatures([domainc.nElX domainc.nElY]);
elseif strcmp(mode, 'useLocal')
    Phi = Phi.localTheta_c([domainc.nElX domainc.nElY]);
end

%% Sample from p_c
disp('Sampling from p_c...')
nTest = testSample_up - testSample_lo + 1;
Xsamples = zeros(domainc.nEl, nSamples_p_c, nTest);
LambdaSamples{1} = zeros(domainc.nEl, nSamples_p_c);
LambdaSamples = repmat(LambdaSamples, nTest, 1);
meanEffCond = zeros(domainc.nEl, nTest);

theta_c.useNeuralNet = false;
if theta_c.useNeuralNet
    finePerCoarse = [sqrt(size(Phi.xk{1}, 1)), sqrt(size(Phi.xk{1}, 1))];
    xkNN = zeros(finePerCoarse(1), finePerCoarse(2), 1, nTest*domainc.nEl);
    k = 1;
    for i = 1:nTest
        for j = 1:domainc.nEl
            xkNN(:, :, 1, k) =...
                reshape(Phi.xk{i}(:, j), finePerCoarse(1), finePerCoarse(2)); %for neural net
            k = k + 1;
        end
    end
end


for i = 1:nTest
    if theta_c.useNeuralNet
        PhiMat = xkNN(:, :, 1, ((i - 1)*domainc.nEl + 1):(i*domainc.nEl));
        mu = double(predict(theta_c.theta, PhiMat));
        Xsamples(:, :, i) = mvnrnd(mu, theta_c.Sigma, nSamples_p_c)';
    else
        Xsamples(:, :, i) = mvnrnd(Phi.designMatrices{i}*theta_c.theta, theta_c.Sigma, nSamples_p_c)';
    end
    LambdaSamples{i} = conductivityBackTransform(Xsamples(:, :, i), condTransOpts);
    if(strcmp(condTransOpts.transform, 'log') && ~theta_c.useNeuralNet)
        meanEffCond(:, i) = exp(Phi.designMatrices{i}*theta_c.theta + .5*diag(theta_c.Sigma));
    else
        meanEffCond(:, i) = mean(LambdaSamples{i}, 2);
    end
end
disp('done')

%% Run coarse model and sample from p_cf
disp('Solving coarse model and sample from p_cf...')
addpath('./heatFEM')
Tf_mean{1} = zeros(domainf.nNodes, 1);
Tf_mean = repmat(Tf_mean, nTest, 1);
TfVarArray = Tf_mean;
Tf_sq_mean = Tf_mean;
parfor j = 1:nTest
    for i = 1:nSamples_p_c
        D = zeros(2, 2, domainc.nEl);
        for e = 1:domainc.nEl
            D(:, :, e) = LambdaSamples{j}(e, i)*eye(2);
        end
        FEMout = heat2d(domainc, D);
        Tctemp = FEMout.Tff';
        
        %sample from p_cf
        mu_cf = theta_cf.mu + theta_cf.W*Tctemp(:);
        %only for diagonal S!!
        %Sequentially compute mean and <Tf^2> to save memory
        Tf_mean{j} = ((i - 1)/i)*Tf_mean{j} + (1/i)*mu_cf;  %U_f-integration can be done analytically
        Tf_sq_mean{j} = ((i - 1)/i)*Tf_sq_mean{j} + (1/i)*mu_cf.^2;
    end
    Tf_sq_mean{j} = Tf_sq_mean{j} + theta_cf.S;
    Tf_var = abs(Tf_sq_mean{j} - Tf_mean{j}.^2);  %abs to avoid negative variance due to numerical error
    meanTf_meanMCErr = mean(sqrt(Tf_var/nSamples_p_c))
    TfVarArray{j} = Tf_var;
    
    meanMahaErrTemp{j} = mean(sqrt((.5./(Tf_var)).*(Tf(:, j) - Tf_mean{j}).^2));
    sqDist{j} = (Tf(:, j) - Tf_mean{j}).^2;
    meanSqDistTemp{j} = mean(sqDist{j})
end
Tf_mean_tot = mean(cell2mat(Tf_mean'), 2);
Tf_sq_mean_tot = mean(cell2mat(Tf_sq_mean'), 2);
meanMahaErr = mean(cell2mat(meanMahaErrTemp));
meanSqDist = mean(cell2mat(meanSqDistTemp));
meanSqDistSq = mean(cell2mat(meanSqDistTemp).^2);
meanSqDistErr = sqrt((meanSqDistSq - meanSqDist^2)/nTest);
rmpath('./rom')
rmpath('./heatFEM')

end
