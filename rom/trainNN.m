function [net] = trainNN(inputData, outputData, finePerCoarse, mode, net)
%Trains a neural net as a model for the effective log conductivity X

if strcmp(mode, 'CNN')
    %set constant seed
    rng(1);
    %define layers
    if nargin > 4
        layers = net.Layers;
    else
%         layers = [imageInputLayer([finePerCoarse 1]), convolution2dLayer(3, 3, 'WeightL2Factor', 1, 'BiasL2Factor', 1), ...
%             reluLayer, fullyConnectedLayer(1, 'WeightL2Factor', 1, 'BiasL2Factor', 1), regressionLayer];
        layers = [imageInputLayer([finePerCoarse 1]), maxPooling2dLayer(3), ...
              fullyConnectedLayer(1, 'WeightL2Factor', 0, 'BiasL2Factor', 0), regressionLayer];
    end
    %Set training options
    options = trainingOptions('sgdm','InitialLearnRate',2e-3, ...
        'MaxEpochs', 300, 'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.95);
    
    tic
    net = trainNetwork(inputData, outputData, layers, options);
    train_time = toc
    
    selfPred = true;
    if selfPred
        disp('Test network on training data...')
        outPred = predict(net, inputData);
        % outPred
        % outputData
        figure
        plot(1:length(outputData), outputData, 'b', 1:length(outputData), outPred, 'r')
        axis tight
        drawnow
        sqDiff = mean((outputData - outPred).^2)
        sqDiffRel = sqDiff/mean(outputData.^2)
    end
elseif strcmp(mode, 'feedforward')
    if nargin > 4
        net = init(net);
    else
        net = fitnet(4, 'trainrp');
    end
    net.divideMode = 'none';
    net.trainParam.epochs = 1e4;
    net.trainParam.min_grad = 1e-5;
    tic
    net = train(net, inputData, outputData);
    train_time = toc
    y = net(inputData);
    err = (1/length(y))*(sqrt(sum((y - outputData)./outputData).^2))
    figure
    plot(1:length(y), y, 1:length(y), outputData)
    axis tight
else
    error('unknown type of neural network')
end
end

