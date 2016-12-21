function [Tf_meanArray, Tf_varArray, Tf_mean_tot, Tf_sq_mean_tot, meanMahaErr, meanSqDist, sqDist] =...
    predictOutput(nSamples_p_c, testSample_lo, testSample_up, testFilePath, modelParamsFolder)
%Function to predict finescale output from generative model

%Load test file
Tffile = matfile(testFilePath);
if nargout > 4
    Tf = Tffile.Tf(:, testSample_lo:testSample_up);
end
[theta_c, theta_cf, domainc, domainf, phi, featureFunctionAbsMean] = loadTrainedParams(modelParamsFolder);

addpath('./rom')

%% Compute design matrices
Phi = DesignMatrix([domainf.nElX domainf.nElY], [domainc.nElX domainc.nElY], phi, Tffile, testSample_lo:testSample_up);
Phi = Phi.computeDesignMatrix(domainc.nEl, domainf.nEl);
%Normalize design matrices
Phi = Phi.normalizeDesignMatrix(featureFunctionAbsMean);

tic;
%% Sample from p_c
disp('Sampling from p_c...')
nTest = testSample_up - testSample_lo + 1;
Xsamples = zeros(domainc.nEl, nSamples_p_c, nTest);
LambdaSamples = Xsamples;
for i = 1:nTest
    Xsamples(:, :, i) = mvnrnd(Phi.designMatrices{i}*theta_c.theta, (theta_c.sigma^2)*eye(domainc.nEl), nSamples_p_c)';
    LambdaSamples(:, :, i) = exp(Xsamples(:, :, i));
end
LambdaSamples(LambdaSamples < 1) = 1;
LambdaSamples(LambdaSamples > 100) = 100;
disp('done')

%% Run coarse model and sample from p_cf
disp('Solving coarse model and sample from p_cf...')
addpath('./heatFEM')
meanMahaErr = 0;
meanSqDist = 0;
sqDist = 0;
Tf_meanArray = zeros(domainf.nNodes, nTest);
Tf_varArray = Tf_meanArray;
%over all training data samples
Tf_mean_tot = zeros(domainf.nNodes, 1);
Tf_sq_mean_tot = zeros(domainf.nNodes, 1);
for j = 1:nTest
    Tf_mean = zeros(domainf.nNodes, 1);
    Tf_sq_mean = zeros(domainf.nNodes, 1);
    for i = 1:nSamples_p_c
        D = zeros(2, 2, domainc.nEl);
        for e = 1:domainc.nEl
            D(:, :, e) = LambdaSamples(e, i, j)*eye(2);
        end
        FEMout = heat2d(domainc, D);
        Tctemp = FEMout.Tff';
        
        %sample from p_cf
        mu_cf = theta_cf.mu + theta_cf.W*Tctemp(:);
        %only for diagonal S!!
        %Sequentially compute mean and <Tf^2> to save memory
        Tf_temp = normrnd(mu_cf, theta_cf.S);
        Tf_mean = ((i - 1)/i)*Tf_mean + (1/i)*Tf_temp;
        Tf_mean_tot = ((i - 1)/i)*Tf_mean_tot + (1/i)*Tf_temp;
        Tf_sq_mean = ((i - 1)/i)*Tf_sq_mean + (1/i)*(Tf_temp.^2);
        Tf_sq_mean_tot = ((i - 1)/i)*Tf_sq_mean_tot + (1/i)*(Tf_temp.^2);
    end
    disp('done')
    Tf_var = abs(Tf_sq_mean - Tf_mean.^2);  %abs to avoid negative variance due to numerical error
    meanTf_meanMCErr = mean(sqrt(Tf_var/nSamples_p_c))
    Tf_meanArray(:, j) = Tf_mean;
    Tf_varArray(:, j) = Tf_var;
    pure_prediction_time = toc
    
    if nargout > 4
        meanMahaErrTemp = mean(sqrt((.5./(Tf_var)).*(Tf(:, j) - Tf_mean).^2));
        meanSqDistTemp = mean(((Tf(:, j) - Tf_mean)./Tf(:, j)).^2);
        sqDistTemp = (Tf(:, j) - Tf_mean).^2;
        meanMahaErr = ((j- 1)/j)*meanMahaErr + (1/j)*meanMahaErrTemp;
        meanSqDist = ((j - 1)/j)*meanSqDist + (1/j)*meanSqDistTemp;
        sqDist = ((j - 1)/j)*sqDist + (1/j)*sqDistTemp;
    end
end
rmpath('./rom')
rmpath('./heatFEM')

end