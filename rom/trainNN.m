function [net] = trainNN(inputData, outputData, finePerCoarse)
%Trains a neural net as a model for the effective log conductivity X

%set constant seed
% rng(1);
%define layers
layers = [imageInputLayer([finePerCoarse 1]), convolution2dLayer(2, 3, 'WeightL2Factor', 1, 'BiasL2Factor', 1), ...
    reluLayer, fullyConnectedLayer(1, 'WeightL2Factor', 1, 'BiasL2Factor', 1), regressionLayer];

%Set training options
options = trainingOptions('sgdm','InitialLearnRate',0.0005, ...
    'MaxEpochs', 100);

tic
net = trainNetwork(inputData, outputData, layers, options);
train_time = toc

selfPred = false;
if selfPred
    disp('Test network on training data...')
    outPred = predict(net, inputData);
    % outPred
    % outputData
    figure
    plot(1:length(outputData), outputData, 'b', 1:length(outputData), outPred, 'r')
    drawnow
    diff = mean(mean(abs(((outputData - outPred)./outputData))))
end
end

