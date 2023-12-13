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
net = fitnet([60,30], 'trainFcn', 'trainscg');
net.trainParam.max_fail = 100;
for i = 1:folds
    XTrain = table2array(digitTable(training(c, i),:));
    YTrain = labels(training(c, i));
    XTest = table2array(digitTable(test(c, i),:));
    YTest = labels(test(c, i));

    net = train(net, XTrain', YTrain');

    pred = round(sim(net, XTest'));
    accuracy = sum(YTest == pred') / numel(YTest);
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
    
end

