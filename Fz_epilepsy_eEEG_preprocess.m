%%%%%%%%  READ AND LOAD EMPIRICAL EPILEPSY DATA %%%%%%%%

%% %%%% ---- Load and Read the Data ---- %%%% %%

% Define the folder containing the .edf files
folder_path = 'D:\OneDrive - CCLAB\Postdoc\Draft\IJCAI 2025\Real Data\Epilepsy Patients\Ses2AcrossSubject'; % Update with your path

% Get list of all EDF files
edf_files = dir(fullfile(folder_path, '*.edf'));
num_subjects = length(edf_files);

% Define number of rows to extract from each subject
num_timepoints = 344; 

% Initialize a matrix to store all subjects' Fz data
all_Fz_data = [];

% Loop through each subject
for i = 1:num_subjects
    % Get the full file path
    file_path = fullfile(folder_path, edf_files(i).name);
    
    % Read the EEG data
    eeg_data = edfread(file_path);
    
    % Check if "Fz" exists in the timetable
    if ismember('Fz', eeg_data.Properties.VariableNames)
        % Extract the Fz column
        Fz_signal = cell2mat(eeg_data.Fz);
        
        % Ensure we take only the first 344 rows
        if length(Fz_signal) >= num_timepoints
            Fz_signal = Fz_signal(1:num_timepoints);
        else
            fprintf('Warning: Subject %s has fewer than 344 time points. Using available data.\n', edf_files(i).name);
            % Zero-pad if necessary to ensure the same length
            Fz_signal = [Fz_signal; zeros(num_timepoints - length(Fz_signal), 1)];
        end
        
        % Store the extracted Fz data
        all_Fz_data(:, i) = Fz_signal; % Each column corresponds to one subject
        
        % Save each subject's EEG data separately
        save_filename = fullfile(folder_path, sprintf('EEG_Subject%d.mat', i));
        save(save_filename, 'Fz_signal');
        
        fprintf('Processed and saved Fz data for %s\n', edf_files(i).name);
    else
        fprintf('Fz channel not found in %s\n', edf_files(i).name);
    end
end

% Save the full matrix containing all subjects' data
save(fullfile(folder_path, 'all_Fz_data.mat'), 'all_Fz_data');

fprintf('Final matrix size: %d x %d (TimePoints x Subjects)\n', size(all_Fz_data,1), size(all_Fz_data,2));


%% %%%% ----- Remove Zero Columns ----%%%% %%

% Load the previously saved matrix
load(fullfile(folder_path, 'all_Fz_data.mat'), 'all_Fz_data');

% Find columns that are entirely zero
nonzero_columns = any(all_Fz_data, 1); % Logical index of columns with nonzero values

% Remove zero-columns
clean_Fz_data = all_Fz_data(:, nonzero_columns);

% Save the cleaned matrix
save(fullfile(folder_path, 'clean_Fz_data.mat'), 'clean_Fz_data');

% Display the final matrix dimensions
fprintf('Cleaned matrix size: %d x %d (TimePoints x Valid Subjects)\n', size(clean_Fz_data,1), size(clean_Fz_data,2));


%% %%% ---  Filtering and Downsampling  ---%%%% %%

Fs = 1000; % Sampling frequency (adjust based on your data)
low_cutoff = 0.5; % Low cutoff frequency
high_cutoff = 40; % High cutoff frequency
[b, a] = butter(4, [low_cutoff high_cutoff] / (Fs / 2), 'bandpass');

filtered_Fz_data = filtfilt(b, a, clean_Fz_data);
save(fullfile(folder_path, 'filtered_Fz_data.mat'), 'filtered_Fz_data');


% Fs_new = 250; % New sampling frequency   % Only if needed
% downsampled_Fz_data = downsample(filtered_Fz_data, Fs / Fs_new);
% save(fullfile(folder_path, 'downsampled_Fz_data.mat'), 'downsampled_Fz_data');
