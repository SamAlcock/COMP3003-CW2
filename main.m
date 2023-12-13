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
kValues = [4, 5, 6, 7, 8];
folds = 5;

% change folds to relevant K value and then put into loop

% c. Creating NN
net = fitnet([60,30], 'trainFcn', 'trainscg');
net.trainParam.max_fail = 50;
for k = 1:numel(kValues)
 
    for i = 1:kValues(k)
        c = cvpartition(labels, "KFold", kValues(k));
        XTrain = table2array(digitTable(training(c, i),:));
        YTrain = labels(training(c, i));
        XTest = table2array(digitTable(test(c, i),:));
        YTest = labels(test(c, i));
    
        net = train(net, XTrain', YTrain');
    
        pred = round(sim(net, XTest'));
        accuracy = sum(YTest == pred') / numel(YTest); % Number of correct predictions / Total number of labels
        fprintf('Maximum folds: %d, Fold number: %d, Accuracy: %.2f%%\n', kValues(k), i, accuracy * 100); % *100 to display as percentage
        
    end
end

