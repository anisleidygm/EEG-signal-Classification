
%%%%%%% BENCHMARK eEEG CLASSIFIERS: CNN AND RNN %%%%%%%%%%


%% STEP 1: Load EEG Data
folderPath = 'D:\OneDrive - CCLAB\Postdoc\Draft\IJCAI 2025\Real Data\Final Real Data';
load(fullfile(folderPath, 'filtered_Oz_data.mat'));         % Healthy (Oz)    160000 x 16
load(fullfile(folderPath, 'filtered_Fz_data.mat'));         % Pathological (Fz)   344 x 8

healthySignals = filtered_Oz_data;
pathologicalSignals = filtered_Fz_data;

% Truncate to equal length
minTimePoints = min(size(healthySignals, 1), size(pathologicalSignals, 1));
healthySignals = healthySignals(1:minTimePoints, :);
pathologicalSignals = pathologicalSignals(1:minTimePoints, :);

%% STEP 2: Format Data for Deep Learning (1 feature x time)
allSignals = [healthySignals, pathologicalSignals];
labels = [zeros(1, size(healthySignals, 2)), ones(1, size(pathologicalSignals, 2))];

X = cell(length(labels), 1);
for i = 1:length(labels)
    signal = allSignals(:, i);
    X{i} = reshape(signal, 1, []);  % [1 x timeSteps] → 1 feature x 344 time
end
Y = categorical(labels(:));

%% STEP 3: Define Network Architectures
layersCNN = [
    sequenceInputLayer(1, "MinLength", 344)
    convolution1dLayer(5, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    convolution1dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

layersRNN = [
    sequenceInputLayer(1, "MinLength", 344)
    lstmLayer(50, 'OutputMode', 'last')
    dropoutLayer(0.3)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

%% STEP 4: 5-Fold Cross-Validation for Both Models
cv = cvpartition(length(Y), 'KFold', 5);
accCNN = zeros(cv.NumTestSets, 1);
accRNN = zeros(cv.NumTestSets, 1);

for i = 1:cv.NumTestSets
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    XTrain = X(trainIdx);
    YTrain = Y(trainIdx);
    XTest = X(testIdx);
    YTest = Y(testIdx);

    try
        % CNN
        netCNN = trainNetwork(XTrain, YTrain, layersCNN, options);
        YPredCNN = classify(netCNN, XTest);
        accCNN(i) = sum(YPredCNN == YTest) / numel(YTest);
    catch ME
        fprintf('CNN error at fold %d: %s\n', i, ME.message);
        accCNN(i) = NaN;
    end

    try
        % RNN
        netRNN = trainNetwork(XTrain, YTrain, layersRNN, options);
        YPredRNN = classify(netRNN, XTest);
        accRNN(i) = sum(YPredRNN == YTest) / numel(YTest);
    catch ME
        fprintf('RNN error at fold %d: %s\n', i, ME.message);
        accRNN(i) = NaN;
    end
end

fprintf('CNN Mean Accuracy: %.2f%%\n', mean(accCNN, 'omitnan') * 100);
fprintf('RNN Mean Accuracy: %.2f%%\n', mean(accRNN, 'omitnan') * 100);
