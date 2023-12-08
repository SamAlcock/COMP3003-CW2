% Tooboxes: Image Processing Toolbox

clc;
close all;
clear;

% a. Loading data and preprocessing
digitTable = readtable('handwritten_digits.csv', 'Headerlines', 1);
labels = digitTable(:, 65);
digitTable(:, 65) = [];

totalNums = height(digitTable);

reshapedDigitTable = cell(size(digitTable, 1) , 1);

for i = 1 : size (digitTable, 1)
    reshapedDigitTable{i} = reshape(data(i, :), [8, 8])

end

numericDigitTable = table2array(reshapedDigitTable);

for i = 1 : size(reshapedDigitTable, 1)
   

end

imagesc(numericDigitTable);
colormap(gray);

% b. Divide the dataset into training and test datasets, cross-validation
% and k-fold
train = 0.75; % 75% training data

