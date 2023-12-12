% Tooboxes: Image Processing Toolbox
clc;
close all;
clear;

% a. Loading data and preprocessing
digitTable = readtable('handwritten_digits.csv', 'Headerlines', 1);
labels = table2array(digitTable(:, 65));
digitTable(:, 65) = [];

totalNums = height(digitTable);
allDigits = zeros(8,8,totalNums); % 3D matrix for all digits
figure;
for i = 1 : 20 % size (digitTable, 1)
    currCol = reshape(table2array(digitTable(i, :)), 8, 8);
    currCol = currCol';
    allDigits(:, :, i) = currCol;
    if i < 10
        subplot(3, 3, i);
    end
    
    imagesc(currCol);
    colormap(gray);
    colorbar;

end




% b. Divide the dataset into training and test datasets, cross-validation
% and k-fold
trainRatio = 0.75; % 75% training data

folds = 5;

c = cvpartition(labels, "KFold", folds);

% c. Creating NN
layers = [
    imageInputLayer([8 8 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

for i = 1:folds
    train = training(c, i);
    validation = test(c, i);

    trainData = allDigits(:, : , train);
    trainLabels = categorical(labels(train)');

    validationData = allDigits(:, :, validation);
    validationLabels = categorical(labels(validation)');
    
    options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', {validationData, validationLabels}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

    net = trainNetwork(trainData, trainLabels, layers, options);
    predictions = classify(net, validationData);
end

