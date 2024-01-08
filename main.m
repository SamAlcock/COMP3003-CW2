% Tooboxes: Image Processing Toolbox
clc;
close all;
clear;

% a. Loading data and preprocessing
digitTable = readtable('handwritten_digits.csv', 'Headerlines', 1);
totalNums = height(digitTable);
digitTable = digitTable(randperm(totalNums), :);
labels = table2array(digitTable(:, 65));
digitTable(:, 65) = [];


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
kValues = [2, 4];

% c. Creating NN

accuracies = zeros(1, length(kValues));
allPred = zeros(numel(kValues), length(labels));
for k = 1:numel(kValues)
    currAccuracies = zeros(1, kValues(k));
    for i = 1:kValues(k)

        net = patternnet([10, 8, 6],'trainlm'); % 10 8 8
        net.trainParam.max_fail = 100;
        c = cvpartition(labels, "KFold", kValues(k));
        XTrain = table2array(digitTable(training(c, i),:));
        YTrain = labels(training(c, i));
        XTest = table2array(digitTable(test(c, i),:));
        YTest = labels(test(c, i));
        
        net.performFcn = 'mse';
        net = train(net, XTrain', YTrain');

        pred = sim(net, XTest');
        accuracy = sum(YTest == round(pred')) / numel(YTest); % Number of correct predictions / Total number of labels  
        currAccuracies(i) = accuracy;
        fprintf('Maximum folds: %d, Fold number: %d, Accuracy: %.2f%%\n', kValues(k), i, accuracy * 100); % *100 to display as percentage
        
    end
    accuracies(k) = mean(currAccuracies(:));
end

% Plotting average accuracy for each fold
figure;
kValLegend = strings(1, length(kValues));
for i = 1:length(kValues)
    kValLegend(i) = num2str(kValues(i));
end

formatAcc = accuracies(:);
b = bar(formatAcc);
set(gca, 'XTickLabel', kValLegend)
xlabel('Number of Folds');
ylabel('Average Accuracy (%)');
title('Accuracy of Neural Network for different K-Folds');
ylim([0.3, 1]);

% Plotting Confusion Matrix
figure;
c = confusionmat(YTest, round(pred'));
confusionchart(c);
title('Confusion Matrix for Naive Bayes for different K-Folds');

% F1 score

classesNum = size(c, 1);
precision = zeros(1, classesNum);
recall = zeros(1, classesNum);
f1Score = zeros(1, classesNum);

for i = 1:classesNum

    precision(i) = c(i, i) / sum(c(:, i));
    recall(i) = c(i, i) / sum(c(i, :));
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));

end
totalF1ScoreNN = mean(f1Score)

% d. Naive Bayes Classifier
cnbAccuracies = zeros(1, length(kValues));
for k = 1:numel(kValues)
        currAccuraciesCnb = zeros(1, kValues(k));
    for i = 1:kValues(k)
        
        c = cvpartition(labels, "KFold", kValues(k));
        XTrain = table2array(digitTable(training(c, i),:));
        YTrain = labels(training(c, i));
        XTest = table2array(digitTable(test(c, i),:));
        YTest = labels(test(c, i));
    
        cnb = fitcnb(XTrain, YTrain, "DistributionNames", "mn");

        cnbPred = predict(cnb, XTest);

        cnbAccuracy = sum(YTest == round(cnbPred)) / numel(YTest); % Number of correct predictions / Total number of labels
        currAccuraciesCnb(i) = cnbAccuracy;
        fprintf('Maximum folds: %d, Fold number: %d, Accuracy: %.2f%%\n', kValues(k), i, cnbAccuracy * 100); % *100 to display as percentage
        
    end
    cnbAccuracies(k) = mean(currAccuraciesCnb(:));
end

% Plotting average accuracy for each fold
figure;


formatAcc = cnbAccuracies(:);
b = bar(formatAcc);
set(gca, 'XTickLabel', kValLegend)
xlabel('Number of Folds');
ylabel('Average Accuracy (%)');
title('Accuracy of Naive Bayes for different K-Folds');
ylim([0.7, 1]);


% Plotting Confusion Matrix
figure;
c = confusionmat(YTest, round(cnbPred'));
confusionchart(c);
title('Confusion Matrix for Naive Bayes for different K-Folds');

% F1 score

classesNum = size(c, 1);
precision = zeros(1, classesNum);
recall = zeros(1, classesNum);
f1Score = zeros(1, classesNum);

for i = 1:classesNum

    precision(i) = c(i, i) / sum(c(:, i));
    recall(i) = c(i, i) / sum(c(i, :));
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));

end
totalF1ScoreNB = mean(f1Score);

% Plotting F1 scores
figure;
formatF1 = [totalF1ScoreNN, totalF1ScoreNB];
b = bar(formatF1);
f1Legend = {'Neural Network', 'Naive Bayes'};
set(gca, 'XTick', 1:numel(f1Legend), 'XTickLabel', f1Legend);
xlabel('Model');
ylabel('F1 Score');
title('Comparing F1 Score when folds = ', num2str(kValues(end)));
ylim([0, 1]);

