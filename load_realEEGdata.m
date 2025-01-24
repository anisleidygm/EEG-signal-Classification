%% Script to load and put all the real EEG data in a single folder.
% Only the Oz channel was extracted since correspond to occipital activity,
% mainly associated with alpha rhythms, charactristic in healthy subjects

%% ------------------
% Set the path to the main EEG1 folder
mainFolder = 'D:\OneDrive - CCLAB\Postdoc\Draft\IJCAI 2025\Real Data\EEG2';

% Create a folder for the final Oz data
finalDataFolder = fullfile(mainFolder, 'final_data');
if ~isfolder(finalDataFolder)
    mkdir(finalDataFolder);
end

% Get a list of all subfolders in the main folder
subfolders = dir(mainFolder);

% Loop through each subfolder
for i = 1:length(subfolders)
    % Skip '.' and '..' directories
    if ~subfolders(i).isdir || strcmp(subfolders(i).name, '.') || strcmp(subfolders(i).name, '..')
        continue;
    end

    % Check if the subfolder name is numeric (indicating subject folder)
    folderName = subfolders(i).name;
    if isnan(str2double(folderName))
        continue;
    end

    % Construct the path to the 'export' folder
    exportFolder = fullfile(mainFolder, folderName, 'export');

    % Check if the 'export' folder exists
    if ~isfolder(exportFolder)
        disp(['Export folder not found in: ', fullfile(mainFolder, folderName)]);
        continue;
    end

    % Look for the .vhdr file in the 'export' folder
    vhdrFile = dir(fullfile(exportFolder, '*.vhdr'));

    if isempty(vhdrFile)
        disp(['No .vhdr file found in: ', exportFolder]);
        continue;
    end

    % Ensure corresponding .dat and .vmrk files exist
    vhdrFilePath = fullfile(exportFolder, vhdrFile(1).name);
    baseName = vhdrFile(1).name(1:end-5); % Base name without extension
    datFile = fullfile(exportFolder, [baseName, '.dat']);
    vmrkFile = fullfile(exportFolder, [baseName, '.vmrk']);

    if ~isfile(datFile) || ~isfile(vmrkFile)
        disp(['Missing .dat or .vmrk file for: ', vhdrFilePath]);
        continue;
    end

    % Read the EEG data using the .vhdr file
    disp(['Processing subject: ', folderName]);
    hdr = ft_read_header(vhdrFilePath); % Read metadata (header)
    data = ft_read_data(vhdrFilePath); % Read EEG data matrix

    % Extract the Oz channel (row 20)
    OzChannelIndex = 20; % Channel index for Oz (adjust if necessary)
    if size(data, 1) < OzChannelIndex
        disp(['Oz channel (20th row) not found in: ', vhdrFilePath]);
        continue;
    end

    % Extract Oz data and time vector
    OzData = data(OzChannelIndex, :); % Extract Oz channel data
    timeVector = (0:size(OzData, 2)-1) / hdr.Fs; % Generate time vector based on sampling frequency

    % Dynamically create valid variable names
    subjectNumber = str2double(folderName); % Convert folder name to a number
    OzDataVarName = sprintf('S%dOzData', subjectNumber); % Example: "S38OzData"
    timeVectorVarName = sprintf('S%dtimeVector', subjectNumber); % Example: "S38timeVector"

    % Save the Oz data and time vector in the final folder with subject-specific variable names
    outputFileName = fullfile(finalDataFolder, [folderName, '_Oz.mat']);
    eval([OzDataVarName, ' = OzData;']); % Create variable dynamically
    eval([timeVectorVarName, ' = timeVector;']); % Create variable dynamically
    save(outputFileName, OzDataVarName, timeVectorVarName);

end

disp('Processing complete. Oz channel data saved in the "final_data" folder.');
