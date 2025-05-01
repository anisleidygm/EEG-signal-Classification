
%%%%%%% SYNTHETIC EEG DATA ANALYSIS PIPELINE %%%%%%%

% The synthetic data was generated using the thalamocortical DD-NMM (https://github.com/anisleidygm/Brain_Modeling)
% This script processes an extended EEG dataset, including both, heathy and pathological simulated signals.
% The methodology includes augmentation of the signals, features extraction, classification and visualization.

%% STEP 1: Load Data
folderPath = ''; % Folder containing .mat files
fileList = dir(fullfile(folderPath, '*.mat'));
data = struct();

% Dynamically load all .mat files
for i = 1:length(fileList)
    fileName = fileList(i).name;
    signalName = erase(fileName, '.mat'); % Remove file extension
    data.(signalName) = load(fullfile(folderPath, fileName)).(signalName);
end

% Extract healthy and pathological signals
healthySignals = {data.healthy1, data.healthy2, data.healthy3};
pathologicalSignals = {data.seizure1, data.seizure2, data.seizure3};

%% STEP 2: Signal Augmentation
NumAugmentedSignals = 100; % Increased number of augmented signals per class
noiseLevel = 0.1; % Noise level
frequencyShift = 0.5; % Frequency modulation factor

% Augment Healthy signals
HealthyAugmented = [];
for i = 1:length(healthySignals)
    for j = 1:NumAugmentedSignals
        phaseShift = randi([1, length(healthySignals{i})]); % Random phase shift
        freqFactor = 1 + (rand - 0.5) * frequencyShift; % Frequency modulation
        augmentedSignal = circshift(healthySignals{i}, phaseShift) .* freqFactor + noiseLevel * randn(size(healthySignals{i}));
        HealthyAugmented = [HealthyAugmented, augmentedSignal];
    end
end

% Augment Pathological signals
PathologicalAugmented = [];
for i = 1:length(pathologicalSignals)
    for j = 1:NumAugmentedSignals
        phaseShift = randi([1, length(pathologicalSignals{i})]); % Random phase shift
        freqFactor = 1 + (rand - 0.5) * frequencyShift; % Frequency modulation
        augmentedSignal = circshift(pathologicalSignals{i}, phaseShift) .* freqFactor + noiseLevel * randn(size(pathologicalSignals{i}));
        PathologicalAugmented = [PathologicalAugmented, augmentedSignal];
    end
end

% Combine augmented data
AugmentedData = [HealthyAugmented, PathologicalAugmented];
AugmentedLabels = [repmat("Healthy", 1, size(HealthyAugmented, 2)), ...
                   repmat("Pathological", 1, size(PathologicalAugmented, 2))];

Fs = 1000; % Sampling frequency

%% STEP 3: Feature Extraction
% Initialize feature matrix
NumSignals = size(AugmentedData, 2);
NumFeatures = 4; % Delta Power, Theta Power, Alpha Power, Peak Frequency
AugmentedFeatures = zeros(NumSignals, NumFeatures);

% Parameters for pwelch
winLength = 512; % Length of each segment
noverlap = winLength / 2; % Overlap between segments
nfft = 1024; % Number of FFT points

for i = 1:NumSignals
    signal = AugmentedData(:, i); % Extract each signal (column)
    
    % Compute power spectral density (PSD)
    [pxx, f] = pwelch(signal, hamming(winLength), noverlap, nfft, Fs);
    totalPower = bandpower(pxx, f, [0 Fs/2], 'psd'); % Total power
    
    % Band powers (normalized)
    deltaPower = bandpower(pxx, f, [0.5 4], 'psd') / totalPower;  % Delta band power
    thetaPower = bandpower(pxx, f, [4 8], 'psd') / totalPower;    % Theta band power
    alphaPower = bandpower(pxx, f, [8 12], 'psd') / totalPower;   % Alpha band power
    
    % Peak Frequency
    [~, peakIndex] = max(pxx);
    peakFrequency = f(peakIndex); % Frequency corresponding to peak PSD
    
    % Store features
    AugmentedFeatures(i, :) = [deltaPower, thetaPower, alphaPower, peakFrequency];
end

% Remove rows with NaN values
nanRows = any(isnan(AugmentedFeatures), 2);
AugmentedFeatures = AugmentedFeatures(~nanRows, :);
AugmentedLabels = AugmentedLabels(~nanRows);

% Normalize features (zero mean, unit variance)
featureStd = std(AugmentedFeatures);
featureStd(featureStd == 0) = 1; % Avoid division by zero
AugmentedFeatures = (AugmentedFeatures - mean(AugmentedFeatures)) ./ featureStd;

%% STEP 4: Classification
% Dynamically adjust cvpartition to match dataset size
cv = cvpartition(size(AugmentedFeatures, 1), 'KFold', 5);

% Train SVM model
mdl = fitcsvm(AugmentedFeatures, AugmentedLabels, 'KernelFunction', 'linear', 'Standardize', true);

% Perform cross-validation
crossValMdl = crossval(mdl, 'CVPartition', cv);
crossValError = kfoldLoss(crossValMdl);
fprintf('Cross-Validation Accuracy: %.2f%%\n', (1 - crossValError) * 100);

% Split dataset for testing
trainIdx = training(cv, 1);
testIdx = test(cv, 1);
XTrain = AugmentedFeatures(trainIdx, :);
YTrain = AugmentedLabels(trainIdx);
XTest = AugmentedFeatures(testIdx, :);
YTest = AugmentedLabels(testIdx);

% Train and test on one fold
mdlFold = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'Standardize', true);
YPred = predict(mdlFold, XTest);

% Evaluate test accuracy
confMat = confusionmat(YTest, YPred);
testAccuracy = sum(diag(confMat)) / sum(confMat, 'all');
fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);

%% STEP 5: Visualization

if isrow(AugmentedLabels)
    AugmentedLabels = AugmentedLabels';
end

% Scatter plot: Alpha Band Power vs Peak Frequency
figure;
gscatter(AugmentedFeatures(:, 3), AugmentedFeatures(:, 4), AugmentedLabels);
xlabel('Alpha Band Power');
ylabel('Peak Frequency (Hz)');
title('Feature Space: Alpha Band Power vs Peak Frequency');
legend('Healthy', 'Pathological');
grid on;

% Power Spectral Density of Original Signals
[pxxHealthy, f] = pwelch(healthySignals{1}, hamming(winLength), noverlap, nfft, Fs);
[pxxPathological, ~] = pwelch(pathologicalSignals{1}, hamming(winLength), noverlap, nfft, Fs);

figure;
plot(f, 10*log10(pxxHealthy), 'b', 'DisplayName', 'Healthy'); hold on;
plot(f, 10*log10(pxxPathological), 'r', 'DisplayName', 'Pathological');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Power Spectral Density');
legend;
grid on;

% Pairwise scatter plot of all features
figure;
gplotmatrix(AugmentedFeatures, [], AugmentedLabels, 'br', '.', [], 'on', [], ...
    {'Delta Power', 'Theta Power', 'Alpha Power', 'Peak Frequency'});

%% Anomaly Detection and t-SNE Visualization
if isrow(AugmentedLabels)
    AugmentedLabels = AugmentedLabels';
end
% Train One-Class SVM on Healthy data
oneClassSVM = fitcsvm(AugmentedFeatures(AugmentedLabels == "Healthy", :), ...
                      ones(sum(AugmentedLabels == "Healthy"), 1), 'KernelFunction', 'rbf', ...
                      'Nu', 0.05, 'Standardize', true);

% Test on all data
[~, anomalyScores] = predict(oneClassSVM, AugmentedFeatures);

% Calculate threshold based on Healthy data
threshold = mean(anomalyScores(AugmentedLabels == "Healthy")) - 2 * std(anomalyScores(AugmentedLabels == "Healthy"));

% Count anomalies
anomalies = anomalyScores < threshold;
fprintf('Number of anomalies detected: %d\n', sum(anomalies));

% Plot anomaly detection scores
figure;
gscatter(1:length(anomalyScores), anomalyScores, AugmentedLabels);
hold on;
yline(threshold, '--r', 'Threshold', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
xlabel('Sample Index');
ylabel('Anomaly Score');
title('Anomaly Detection Scores');
legend('Healthy', 'Pathological', 'Threshold');
grid on;
hold off;

% Perform t-SNE
tsneResults = tsne(AugmentedFeatures, 'NumDimensions', 2, 'Perplexity', 30);

% Plot t-SNE results
figure;
gscatter(tsneResults(:, 1), tsneResults(:, 2), AugmentedLabels);
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
title('t-SNE Visualization of Healthy and Pathological Features');
legend('Healthy', 'Pathological');
grid on;
