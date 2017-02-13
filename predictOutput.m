function [TfMeanArray, TfVarArray, Tf_mean_tot, Tf_sq_mean_tot, meanMahaErr, meanSqDist, sqDist, meanEffCond, meanSqDistErr] =...
    predictOutput(nSamples_p_c, testSample_lo, testSample_up, testFilePath, modelParamsFolder, useNeighbor)
%Function to predict finescale output from generative model

%Load test file
Tffile = matfile(testFilePath);
if nargout > 4
    Tf = Tffile.Tf(:, testSample_lo:testSample_up);
end
[theta_c, theta_cf, domainc, domainf, phi, featureFunctionMean, featureFunctionSqMean,...
    featureFunctionMin, featureFunctionMax] = loadTrainedParams(modelParamsFolder);

addpath('./rom')
addpath('./aux')

%% Compute design matrices
Phi = DesignMatrix([domainf.nElX domainf.nElY], [domainc.nElX domainc.nElY], phi, Tffile, testSample_lo:testSample_up);
load('./data/conductivityTransformation.mat');
%change here for anisotropy!
condTransOpts.anisotropy = false;
Phi = Phi.computeDesignMatrix(domainc.nEl, domainf.nEl, condTransOpts);
%Normalize design matrices
%Phi = Phi.standardizeDesignMatrix(featureFunctionMean, featureFunctionSqMean);
Phi = Phi.rescaleDesignMatrix(featureFunctionMin, featureFunctionMax);
if useNeighbor
    %use feature function information from nearest neighbors
    Phi = Phi.includeNearestNeighborFeatures([domainc.nElX domainc.nElY]);
end

%% Sample from p_c
disp('Sampling from p_c...')
nTest = testSample_up - testSample_lo + 1;
Xsamples = zeros(domainc.nEl, nSamples_p_c, nTest);
LambdaSamples{1} = zeros(domainc.nEl, nSamples_p_c);
LambdaSamples = repmat(LambdaSamples, nTest, 1);
meanEffCond = zeros(domainc.nEl, nTest);
for i = 1:nTest
    Xsamples(:, :, i) = mvnrnd(Phi.designMatrices{i}*theta_c.theta, (theta_c.sigma^2)*eye(domainc.nEl), nSamples_p_c)';
    LambdaSamples{i} = conductivityBackTransform(Xsamples(:, :, i), condTransOpts);
    if strcmp(condTransOpts.transform, 'log')
        meanEffCond(:, i) = exp(Phi.designMatrices{i}*theta_c.theta + .5*theta_c.sigma^2);
    else
        meanEffCond(:, i) = mean(LambdaSamples{i}, 2);
    end
end
disp('done')

%% Run coarse model and sample from p_cf
disp('Solving coarse model and sample from p_cf...')
addpath('./heatFEM')
TfMeanArray{1} = zeros(domainf.nNodes, 1);
TfMeanArray = repmat(TfMeanArray, nTest, 1);
TfVarArray = TfMeanArray;
%over all training data samples
Tf_mean{1} = zeros(domainf.nNodes, 1);
Tf_mean = repmat(Tf_mean, nTest, 1);
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
        Tf_temp = normrnd(mu_cf, sqrt(theta_cf.S));
        Tf_mean{j} = ((i - 1)/i)*Tf_mean{j} + (1/i)*Tf_temp;
        Tf_sq_mean{j} = ((i - 1)/i)*Tf_sq_mean{j} + (1/i)*(Tf_temp.^2);
    end
    disp('done')
    Tf_var = abs(Tf_sq_mean{j} - Tf_mean{j}.^2);  %abs to avoid negative variance due to numerical error
    meanTf_meanMCErr = mean(sqrt(Tf_var/nSamples_p_c))
    TfMeanArray{j} = Tf_mean{j};
    TfVarArray{j} = Tf_var;
    
    meanMahaErrTemp{j} = mean(sqrt((.5./(Tf_var)).*(Tf(:, j) - Tf_mean{j}).^2));
    sqDistTemp{j} = (Tf(:, j) - Tf_mean{j}).^2;
    meanSqDistTemp{j} = mean(sqDistTemp{j})
end
Tf_mean_tot = mean(cell2mat(TfMeanArray'), 2);
Tf_sq_mean_tot = mean(cell2mat(Tf_sq_mean'), 2);
meanMahaErr = mean(cell2mat(meanMahaErrTemp));
meanSqDist = mean(cell2mat(meanSqDistTemp));
meanSqDistSq = mean(cell2mat(meanSqDistTemp).^2);
meanSqDistErr = sqrt((meanSqDistSq - meanSqDist^2)/nTest);
sqDist = mean(cell2mat(sqDistTemp'), 2);
rmpath('./rom')
rmpath('./heatFEM')

end
