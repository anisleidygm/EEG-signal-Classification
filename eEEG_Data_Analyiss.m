
%%%%%%%  EMPIRICAL EEG DATA ANALYSIS PIPELINE %%%%%%%%%%

% Get the preprocessed empirical data and make some feature exteraction,
% classificatin and anomay detection

%% STEP 1: Load EEG Data
folderPath = 'D:\OneDrive - CCLAB\Postdoc\Draft\IJCAI 2025\Real Data\Final Real Data';

% Load healthy (Oz) and pathological (Fz) EEG data
healthyData = load(fullfile(folderPath, 'filtered_Oz_data.mat'));
pathologicalData = load(fullfile(folderPath, 'filtered_Fz_data.mat'));

% Extract EEG signals
healthySignals = healthyData.filtered_Oz_data; % Healthy (Oz)
pathologicalSignals = pathologicalData.filtered_Fz_data; % Pathological (Fz)

% Ensure both datasets have the same time length
minTimePoints = min(size(healthySignals, 1), size(pathologicalSignals, 1));
healthySignals = healthySignals(1:minTimePoints, :);
pathologicalSignals = pathologicalSignals(1:minTimePoints, :);

% Display dataset sizes
fprintf('Healthy Data Size: %d x %d\n', size(healthySignals, 1), size(healthySignals, 2));
fprintf('Pathological Data Size: %d x %d\n', size(pathologicalSignals, 1), size(pathologicalSignals, 2));

%% STEP 2: Feature Extraction
Fs = 1000; % Sampling frequency
NumSubjects = size(healthySignals, 2) + size(pathologicalSignals, 2);
NumFeatures = 4; % Delta Power, Theta Power, Alpha Power, Peak Frequency
Features = zeros(NumSubjects, NumFeatures);
Labels = [repmat("Healthy", size(healthySignals, 2), 1); repmat("Pathological", size(pathologicalSignals, 2), 1)];

% Parameters for pwelch
winLength = min(512, floor(minTimePoints / 2)); % Ensure window is not too large
noverlap = winLength / 2;
nfft = 1024;

% Extract Features for Each Subject
for i = 1:size(healthySignals, 2)
    signal = healthySignals(:, i);
    [pxx, f] = pwelch(signal, hamming(winLength), noverlap, nfft, Fs);
    if isempty(pxx) || length(f) < 2, continue; end
    
    % Compute normalized band powers
    totalPower = bandpower(pxx, f, [0 Fs/2], 'psd');
    deltaPower = bandpower(pxx, f, [0.5 4], 'psd') / totalPower;
    thetaPower = bandpower(pxx, f, [4 8], 'psd') / totalPower;
    alphaPower = bandpower(pxx, f, [8 12], 'psd') / totalPower;
    [~, peakIndex] = max(pxx);
    peakFrequency = f(peakIndex);
    
    Features(i, :) = [deltaPower, thetaPower, alphaPower, peakFrequency];
end

for i = 1:size(pathologicalSignals, 2)
    signal = pathologicalSignals(:, i);
    [pxx, f] = pwelch(signal, hamming(winLength), noverlap, nfft, Fs);
    if isempty(pxx) || length(f) < 2, continue; end
    
    totalPower = bandpower(pxx, f, [0 Fs/2], 'psd');
    deltaPower = bandpower(pxx, f, [0.5 4], 'psd') / totalPower;
    thetaPower = bandpower(pxx, f, [4 8], 'psd') / totalPower;
    alphaPower = bandpower(pxx, f, [8 12], 'psd') / totalPower;
    [~, peakIndex] = max(pxx);
    peakFrequency = f(peakIndex);
    
    Features(size(healthySignals, 2) + i, :) = [deltaPower, thetaPower, alphaPower, peakFrequency];
end

% Remove NaN or Inf values
validRows = all(~isnan(Features) & ~isinf(Features), 2);
Features = Features(validRows, :);
Labels = Labels(validRows);

% Normalize Features
Features = (Features - mean(Features)) ./ std(Features);

%% STEP 3: Classification with SVM
% Cross-validation setup
cv = cvpartition(size(Features, 1), 'KFold', 5);

% Train SVM model
mdl = fitcsvm(Features, Labels, 'KernelFunction', 'linear', 'Standardize', true);

% Cross-validation accuracy
crossValMdl = crossval(mdl, 'CVPartition', cv);
crossValError = kfoldLoss(crossValMdl);
fprintf('Cross-Validation Accuracy: %.2f%%\n', (1 - crossValError) * 100);

% Train-test split
trainIdx = training(cv, 1);
testIdx = test(cv, 1);
XTrain = Features(trainIdx, :);
YTrain = Labels(trainIdx);
XTest = Features(testIdx, :);
YTest = Labels(testIdx);

% Train and test the model
mdlFold = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'Standardize', true);
YPred = predict(mdlFold, XTest);

% Compute test accuracy
confMat = confusionmat(YTest, YPred);
testAccuracy = sum(diag(confMat)) / sum(confMat, 'all');
fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);

%% STEP 4: Anomaly Detection with Anomaly Score
healthyIdx = Labels == "Healthy";
healthyFeatures = Features(healthyIdx, :);

% Train One-Class SVM on Healthy Data
oneClassSVM = fitcsvm(healthyFeatures, ones(sum(healthyIdx), 1), ...
                      'KernelFunction', 'rbf', 'Nu', 0.05, 'Standardize', true);

% Compute anomaly scores for all data
[~, anomalyScores] = predict(oneClassSVM, Features);

% Compute threshold based on healthy subjects
threshold = mean(anomalyScores(healthyIdx)) - 2 * std(anomalyScores(healthyIdx));
anomalies = anomalyScores < threshold;

fprintf('Number of anomalies detected: %d\n', sum(anomalies));

% Plot anomaly scores
figure;
scatter(1:length(anomalyScores), anomalyScores, 50, 'filled');
hold on;
yline(threshold, '--r', 'Threshold', 'LineWidth', 2);
xlabel('Sample Index');
ylabel('Anomaly Score');
title('Anomaly Detection Scores');
legend('Subjects', 'Threshold');
grid on;
hold off;

%% STEP 5: t-SNE Visualization with Anomalies
numSamples = size(Features, 1);
adjustedPerplexity = min(30, floor(numSamples / 3));

tsneResults = tsne(Features, 'NumDimensions', 2, 'Perplexity', adjustedPerplexity);

figure;
gscatter(tsneResults(:, 1), tsneResults(:, 2), Labels);
hold on;
scatter(tsneResults(anomalies, 1), tsneResults(anomalies, 2), 100, 'kx', 'LineWidth', 2);
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
title('t-SNE Visualization with Anomalies');
legend('Healthy', 'Pathological', 'Anomalies');
grid on;
hold off;
