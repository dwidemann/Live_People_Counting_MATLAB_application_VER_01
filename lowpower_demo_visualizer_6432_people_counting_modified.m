c_light = 299792458; % m/s
if exist('outsideRun','var')
    if(~outsideRun)
        clear, clc;
    end
end
close(findall(0, 'type', 'figure'))
cd(fileparts(which(mfilename)))

gui = 0;
if exist('outsideRun','var')
    if(outsideRun)
        gui = 0;
    end
end

if gui == 1
    % Specify the file name
    filename = 'load.txt';
    if exist(filename, 'file')
        % Specify the file path
        filepath = 'load.txt';  % Replace with the actual path
        
        % Read the lines from the file
        try
            lines = textread(filepath, '%s', 'delimiter', '\n');
        catch
            error('Error reading the file. Make sure the file exists and the path is correct.');
        end
    else
        lines{1} = '';
        lines{2} = '';
        lines{3} = '';
    end

    [file1, file2, folder, fig] = fileSelector(lines);
    if strcmp(folder,'')
        folder{1} = '';
    end
    close(fig);
    
    fileID = fopen(filename, 'w');  % 'w' for write
    
    % Write your content (e.g., header)
    fprintf(fileID, strcat(strrep(file1{1},'\','/'),'\n'));
    fprintf(fileID, strcat(strrep(file2{1},'\','/'),'\n'));
    fprintf(fileID, strcat(strrep(folder{1},'\','/'),'\n'));
    
    % Close the file
    fclose(fileID);
else
    file1{1} = 'cfg_file_application_4m_straight_troublemakers_tv.cfg';
    file2{1} = 'params.txt';
    folder{1} = strcat('./dataHeatmap/',datestr(today, 'yyyymmdd'));
end

chirpConfigurationFileName = file1{1};
% UART speeds
controlPortBaudrate = 115200;
dataPortBaudqrate    = 115200;

% Serial Ports
controlSerialPort = 7;
dataSerialPort = 7;

numFramesToAccumulate = 10;
heatmapCollapseMethod = 1; %1: Average 2: Max
peopleCountingBufferSize = 40;
rMaxCombination = 30;
true_num_people = 1;

% thresholds:
hmin1 = 100; % hmin transform
hmin2 = 20; % hmin transform
otsuMultiplier = 0.95; %multiplier for Otsu Threshold
minPeakValue = 250; %minimum value to be considered as a peak
minPoints = 5; %min points to be considered as a blob

saveBool = 1;
saveDataFrames = 240;
heatmapsToSave = [];
room = 'LV-1F-C7';
fHist = struct;
% Define the folder path
folderPath = folder{1};

playBack = 0;

if exist('outsideRun','var')
    if(outsideRun)
        playBack = 1;
    end
end
if playBack == 1
    gui = 0;
    saveBool = 0;
    playBackFile = "C:\Users\a0491005\Documents\TI\lowpowerpeoplecounting\PeopleCountingExperimentsGaurang\dataHeatmap\data_June2024_Anushree1DCapon\5\fHist_adc_data_0008.mat";
end

params = parseFile(file2{1});
if size(params,2) ~= 0
    controlSerialPort = params{1,1}{1,2};
    dataSerialPort = params{1,1}{1,2};

    numFramesToAccumulate = params{1,2}{1,2};
    heatmapCollapseMethod = params{1,2}{1,3}; 
    peopleCountingBufferSize = params{1,2}{1,4};
    true_num_people = params{1,2}{1,5};
    rMaxCombination = params{1,2}{1,6};
    hmin1 = params{1,2}{1,7}; % hmin transform
    hmin2= params{1,2}{1,8}; % hmin transform
    otsuMultiplier = params{1,2}{1,9}; %multiplier for Otsu Threshold
    minPoints = params{1,2}{1,10}; %min points to be considered as a blob

    saveBool = params{1,3}{1,2};
    saveDataFrames = params{1,3}{1,3};
    room = params{1,3}{1,4};
end

if exist('outsideRun','var')
    if(outsideRun)
        playBackFile = testFile;
        true_num_people = class;
    end
end

accuracy = 0;
accuracy_mean = 0;
accuraccyGreaterThan = 0;       %Considers all numbers <= num_people_frame to be equal to num_people_frame

if exist('outsideRun','var')
    [fig, sensitivity] = createSliderFigure(outsideRun);
else
    [fig, sensitivity] = createSliderFigure(0);
end

if playBack == 0
%% TLV CODES
    % TLV types
    MMWDEMO_OUTPUT_EXT_MSG_RANGE_FFT                        = 321;
    
    syncPatternUINT64 = typecast(uint16([hex2dec('0102'),hex2dec('0304'),hex2dec('0506'),hex2dec('0708')]),'uint64');
    syncPatternUINT8 = typecast(uint16([hex2dec('0102'),hex2dec('0304'),hex2dec('0506'),hex2dec('0708')]),'uint8');
        
    frameHeaderStructType = struct(...
        'sync',             {'uint64', 8}, ... % See syncPatternUINT64 below
        'version',          {'uint32', 4}, ... % 
        'packetLength',     {'uint32', 4}, ... % In bytes, including header
        'platform',         {'uint32', 4}, ... %
        'frameNumber',      {'uint32', 4}, ... % Starting from 1
        'timeinCPUcycles',  {'uint32', 4}, ... % Not used
        'numDetectObject',  {'uint32', 4}, ... % 
        'numTLVs' ,         {'uint32', 4}, ... % Number of TLVs in this frame
        'subframe',         {'uint32', 4});    % Header checksum
    
    tlvHeaderStruct = struct(...
        'type',                 {'uint32', 4}, ... % TLV object Type
        'length',               {'uint32', 4});    % TLV object Length, in bytes, including TLV header
        
    % Point Cloud TLV reporting unit for all reported points
    pointUintStruct = struct(...
        'xyzUnit',              {'float', 4}, ... % x/y/z in m
        'dopplerUnit',          {'float', 4}, ... % Doplper, in m/s
        'snrUnit',              {'float', 4}, ... % SNR, in dB
        'noiseUnit',            {'float', 4}, ... % noise, in dB
        'numDetPointsMajor',    {'uint16', 2}, ... % number of detected points in Major mode
        'numDetPointsMinor',    {'uint16', 2});    % number of detected points in Minor mode
        
    % Point Cloud TLV object consists of an array of points.
    % Each point has a structure defined below
    pointStructCartesian = struct(...
        'x',                    {'int16', 2}, ... % x in m
        'y',                    {'int16', 2}, ... % y in m
        'z',                    {'int16', 2}, ... % z in m
        'doppler',              {'int16', 2}, ... % Doplper, in m/s
        'snr',                  {'uint8', 1},...    % SNR ratio in 0.1dB
        'noise',                {'uint8', 1});    % type 0-major motion, 1-minor motion
    
    pointStructXYZ = struct(... 
        'x', {'float',4}, ...   % X 
        'y', {'float',4}, ...   % Y
        'z', {'float',4}, ...   % z
        'doppler', {'float',4}); %doppler 
        
    % Side info TLV 7 - MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO
    pointStructXYZ_sideinfo = struct(... 
        'snr', {'uint16',2}, ...   % SNR 
        'noise', {'uint16',2}); %Noise 
        
    frameHeaderLengthInBytes = lengthFromStruct(frameHeaderStructType);
    tlvHeaderLengthInBytes = lengthFromStruct(tlvHeaderStruct);
end
%% Configurations
% Read Chirp Configuration file
cliCfg = readCfg(chirpConfigurationFileName);
Params = parseCfg6432(cliCfg);

% Is Rx channel compensation enabled
if isfield(Params,'measureRangeBiasAndRxChanPhase')
    rxChainCompensationEnabled = Params.measureRangeBiasAndRxChanPhase.enabled;
else
    rxChainCompensationEnabled = 0;
end
enableMajorMotion = Params.sigProcChainCfg.enableMajorMotion;
enableMinorMotion = Params.sigProcChainCfg.enableMinorMotion;
highLevelCliCommands = {'mpdBoundaryBox', 'sensorPosition', 'majorStateCfg', 'minorStateCfg', 'clusterCfg'};
%% Scene and Sensor Parameters
% Rotation parameters
scene.elevationTilt = Params.sensorPosition.elevationTilt*pi/180;
scene.azimuthTilt = Params.sensorPosition.azimuthTilt*pi/180;
scene.sensorPos = [Params.sensorPosition.xOffset Params.sensorPosition.yOffset Params.sensorPosition.zOffset];
azimuthTilt = scene.azimuthTilt;
elevationTilt = scene.elevationTilt;
scene.RotZ_TW = [cos(azimuthTilt) sin(azimuthTilt) 0; -sin(azimuthTilt) cos(azimuthTilt) 0; 0 0 1];
scene.RotX_TW = [1 0 0; 0 cos(elevationTilt) sin(elevationTilt); 0 -sin(elevationTilt) cos(elevationTilt)];

% Sensor parameters
sensor.rangeMax = Params.dataPath.rangeResolutionMeters*Params.dataPath.numRangeBins;
sensor.rangeMin = 1;
sensor.azimFoV = 140*pi/180; %+/-70 degree FOV in azimuth direction
sensor.elevFoV = 60*pi/180; %+/-30 degree FOV in elevation direction
sensor.framePeriod = Params.frameCfg.framePeriodicity; %in ms
sensor.rangeResolution = Params.dataPath.rangeResolutionMeters;
sensor.maxRadialVelocity = Params.dataPath.dopplerResolutionMps*Params.frameCfg.numLoops/2;
sensor.radialVelocityResolution = Params.dataPath.dopplerResolutionMps;
sensor.azim = linspace(-sensor.azimFoV/2, sensor.azimFoV/2, 128);
sensor.elev = linspace(-sensor.elevFoV/2, sensor.elevFoV/2, 128);

%% Initialize variables and structures
lostSync = 0;
gotHeader = 0;
bytesAvailableMax = 0;
rngProfileMajor = [];
adcSamples = [];

frameNum = 1;
targetFrameNum = 0;
peopleCountBuffer = zeros(peopleCountingBufferSize,1) - 1;
fprintf('------------------\n');

sigProcCfg = split(cliCfg{1,8});
rangeSelCfg = split(cliCfg{1,11});
maxRange = str2num(rangeSelCfg{3});
numFrmPerMinorMotProc = str2num(sigProcCfg{6});
if Params.channelCfg.txChannelEn == 3
    numTxAntEn = 2;
else
    numTxAntEn = 1;
end

if Params.channelCfg.rxChannelEn == 7
    numRxAntEn = 3;
else
    numRxAntEn = 2;
end
rangeFFT = zeros(numFrmPerMinorMotProc, Params.frameCfg.numOfChirpsInBurst*Params.frameCfg.numOfBurstsInFrame/numTxAntEn * numTxAntEn*numRxAntEn * Params.chirpComnCfg.numAdcSamples/2);

% Create steering vector grids with FFT-based method
% antenna distance over lambda
antennaGeometryCfg = split(cliCfg{1,6});
%lambda = physconst('LightSpeed')/(Params.chirpTimingCfg.startFreq*1e9 + (Params.chirpComnCfg.numAdcSamples/(Params.chirpComnCfg.digOutputSampRate*1e3)/2)*Params.chirpTimingCfg.chirpSlope*1e12);
lambda = c_light/(Params.chirpTimingCfg.startFreq*1e9 + (Params.chirpComnCfg.numAdcSamples/(Params.chirpComnCfg.digOutputSampRate*1e3)/2)*Params.chirpTimingCfg.chirpSlope*1e12);
d_over_lambda = str2num(antennaGeometryCfg{14})/lambda;
FFTsize_Angle_Az = str2num(sigProcCfg{2});
FFTsize_Angle_El = str2num(sigProcCfg{3});
numChirps = (Params.frameCfg.numOfChirpsInBurst/numTxAntEn * (Params.frameCfg.chirpEndIdx - Params.frameCfg.chirpStartIdx +1));

combinedHeatmaps = zeros(Params.chirpComnCfg.numAdcSamples/2,FFTsize_Angle_Az,numFramesToAccumulate);

a1 = (-FFTsize_Angle_Az/2 : FFTsize_Angle_Az/2-1) / d_over_lambda / FFTsize_Angle_Az;
a2 = (-FFTsize_Angle_El/2 : FFTsize_Angle_El/2-1) / d_over_lambda / FFTsize_Angle_El;
A = zeros(numTxAntEn*numRxAntEn, length(a1)*length(a2));
m_ind               =   -[0, 1, 2, 1, 2, 3];
n_ind               =   -[1, 0, 1, 1, 0, 1];

for ii = 1:length(a2)
    A(:, (ii-1)*length(a1)+1:ii*length(a1))     =   exp(-1j*2*pi*d_over_lambda*(m_ind.'*a1 + n_ind.'*a2(ii)));
end

doaParams = struct('phaseCompArray',[1 -1 1 -1 1 -1]', ...
    'steerMat',A,'m_ind',m_ind,'n_ind',n_ind,'virtualAntToProcess',1:6, ...
    'FFTsize_Doppler',2^ceil(log2(numChirps/numTxAntEn)), ...
    'FFTsize_Angle_Az',FFTsize_Angle_Az,'FFTsize_Angle_El',FFTsize_Angle_El,'cov_numChirps',numChirps/numTxAntEn, 'numFrmPerMinorMotProc',numFrmPerMinorMotProc);

[~,boundaryPoints] = getBoundaryPoints(FFTsize_Angle_Az, Params);

if playBack == 0
    %% UART Cfg file
    % Configure control UART port
    hControlSerialPort = configureControlPort(controlSerialPort,controlPortBaudrate);
          
    % Send CLI configuration
    fprintf('Sending configuration from %s file to Target EVM ...\n', chirpConfigurationFileName);
    for k=1:length(cliCfg)
        if isempty(strrep(strrep(cliCfg{k},char(9),''),char(32),''))
            continue;
        end
        if strcmp(cliCfg{k}(1),'%')
            continue;
        end
        if any(strcmp(highLevelCliCommands,strtok(cliCfg{k},' ')))
            continue;
        end
        
        if controlPortBaudrate > 115200
            writelineslow(hControlSerialPort, cliCfg{k});
        else
            writelineslow(hControlSerialPort, cliCfg{k});
        end
        fprintf('%s\n', cliCfg{k});
    
        if strcmp('baudRate',strtok(cliCfg{k},' '))
            [~, baudRate] = strtok(cliCfg{k},' ');
            set(hControlSerialPort, 'BaudRate', str2double(baudRate));
            controlPortBaudrate = str2double(baudRate);
            clear baudRate;
            pause(.5);
            continue;
        end
    
        for kk = 1:3
            cc = readline(hControlSerialPort);
            if contains(cc, 'Done')
                fprintf('%s\n',cc);
                break;
            elseif contains(cc, 'not recognized as a CLI command')
                fprintf('%s\n',cc);
            elseif contains(cc, 'Debug:')
                fprintf('%s\n',cc);
            elseif contains(cc, 'Error')
                fprintf('%s\n',cc);
                return;
            end
        end
    end
    
    % Reconfigure data UART port as control UART port
    hDataSerialPort = reconfigureControlPort(hControlSerialPort);
else
    fHistfile = load(playBackFile);
end

%%
while(1)
    % Frame Processing in this loop
    while(lostSync == 0)
        dt = datestr(datetime('now','TimeZone', 'local'));
        unix_time = now;
%% Read data from UART
        if playBack == 0
            if(gotHeader == 0)
                % Read the header first
                [rxHeader, byteCount, outOfSyncBytes] = readFrameHeader(hDataSerialPort, frameHeaderLengthInBytes, syncPatternUINT8);
                gotHeader = 1;
            end
    
            % Double check the header size
            if(byteCount ~= frameHeaderLengthInBytes)
                reason = 'Header Size is wrong';
                lostSync = 1;
                break;
            end
    
            % Double check the sync pattern
            magicBytes = typecast(uint8(rxHeader(1:8)), 'uint64');
            if(magicBytes ~= syncPatternUINT64)
                reason = 'No SYNC pattern';
                lostSync = 1;
                break;
            end
    
            % Update the bytes available
            bytesAvailable = hDataSerialPort.NumBytesAvailable;
            if(bytesAvailable > bytesAvailableMax)
                bytesAvailableMax = bytesAvailable;
            end
            
            % parse the header
            frameHeader = readToStruct(frameHeaderStructType, rxHeader);
            if(gotHeader == 1)
                if(frameHeader.frameNumber >= targetFrameNum)
                    % We have a valid header
                    targetFrameNum = frameHeader.frameNumber;
                    if outOfSyncBytes > 0
                        disp(['Found sync at frame ',num2str(targetFrameNum),'(',num2str(frameNum),'). ', 'Discarded out of sync bytes: ' num2str(outOfSyncBytes)]);
                    end
                    gotHeader = 0;
                else
                    reason = 'Old Frame';
                    gotHeader = 0;
                    lostSync = 1;
                    break;
                end
            end
    
            dataLength = frameHeader.packetLength - frameHeaderLengthInBytes;
    
            if(dataLength > 0)
                % Read all packet
                [rxData, byteCount] = readSerial(hDataSerialPort, double(dataLength), 'uint8');
                if(byteCount ~= double(dataLength))
                    reason = 'Data Size is wrong'; 
                    lostSync = 1;
                    break;  
                end
                offset = 0;
    
                % TLV Parsing
                for nTlv = 1:frameHeader.numTLVs
                    tlvType = typecast(uint8(rxData(offset+1:offset+4)), 'uint32');
                    tlvLength = typecast(uint8(rxData(offset+5:offset+8)), 'uint32');
                    if(tlvLength + offset > dataLength)
                        reason = 'TLV Size is wrong';
                        lostSync = 1;
                        break;                    
                    end
                    offset = offset + tlvHeaderLengthInBytes;
                    valueLength = tlvLength;
    
                    if tlvType == MMWDEMO_OUTPUT_EXT_MSG_RANGE_FFT   
                        rngProfileMajor = double(typecast(uint8(rxData(offset+1: offset+valueLength)),'int16'));
                        rngProfileMajor = rngProfileMajor(1,2:2:end) + 1i*rngProfileMajor(1,1:2:end);
                        if frameNum <= numFrmPerMinorMotProc
                            rangeFFT(frameNum,:) = rngProfileMajor;
                        else
                            rangeFFTPrev = rangeFFT;
                            rangeFFT(1:numFrmPerMinorMotProc-1,:) = rangeFFT(2:numFrmPerMinorMotProc,:);
                            rangeFFT(numFrmPerMinorMotProc,:) = rngProfileMajor;
                        end
                        offset = offset + valueLength;         
                    else
                        reason = 'TLV Type is wrong';
                        lostSync = 1;
                        break;     
                    end
                end
                if(lostSync)
                    break;
                end
            end
            disp(['Received data from frame ', num2str(frameNum)]);
        end
        frameNum = frameNum + 1;   

    %% Processing into RA heatmap
    if playBack == 0
        if frameNum <= numFrmPerMinorMotProc
            continue;
        end
        
        rangeProfile = zeros(Params.chirpComnCfg.numAdcSamples/2,numTxAntEn*numRxAntEn,numFrmPerMinorMotProc*Params.frameCfg.numOfChirpsInBurst*Params.frameCfg.numOfBurstsInFrame/numTxAntEn);
        for i = 1:size(rangeFFT,1)
            rangeFFTFrame = reshape(rangeFFT(i,:),[Params.chirpComnCfg.numAdcSamples/2 numTxAntEn*numRxAntEn Params.frameCfg.numOfChirpsInBurst*Params.frameCfg.numOfBurstsInFrame/numTxAntEn]);
            rangeProfile(:,:,(i-1)*Params.frameCfg.numOfChirpsInBurst*Params.frameCfg.numOfBurstsInFrame/numTxAntEn+1:i*Params.frameCfg.numOfChirpsInBurst*Params.frameCfg.numOfBurstsInFrame/numTxAntEn) = rangeFFTFrame;
        end
    
        rangeAngle = doa_rangeAngleHeatmapGen_capon(rangeProfile, doaParams);
    else
%         rangeAngle = flip(heatmapsToSave(:,:,frameNum));
        rangeAngle = fHistfile.fHist(frameNum).rangeAzimuthHeatMapMinor;
    end
    
    % if saveBool == 1
    %     if rem(frameNum,saveDataFrames) == 0
    %         % Check if the folder exists, if not, create it
    %         if ~exist(folderPath, 'dir')
    %             mkdir(folderPath);
    %         end
    % 
    %         % Get a list of .mat files in the folder
    %         files = dir(fullfile(folderPath, '*.mat'));
    % 
    %         % Get the number of existing files
    %         numFiles = length(files);
    % 
    %         % Find the largest file number in the folder
    %         maxFileNum = 0;
    %         for i = 1:numFiles
    %             % Extract the file number from the filename
    %             [~, name, ~] = fileparts(files(i).name);
    %             fileNum = str2double(regexp(name, '\d*', 'match'));
    % 
    %             % Update the maximum file number
    %             if fileNum > maxFileNum
    %                 maxFileNum = fileNum;
    %             end
    %         end
    % 
    %         % Define the next integer y
    %         y = maxFileNum + 1;
    % 
    %         % Define the filename
    %         filename = sprintf('%d.mat', y);
    % 
    %         % Full file path
    %         fullFilePath = fullfile(folderPath, filename);
    % 
    %         % Save 'fHist' to a .mat file
    %         save(fullFilePath, 'fHist');
    %         disp(fprintf('Saved %s',fullFilePath));
    % 
    %         fHist = struct;
    %     else
    %         fHist(rem(frameNum,saveDataFrames)).rangeAzimuthHeatMapMinor = rangeAngle;
    %         fHist(rem(frameNum,saveDataFrames)).room = room;
    %         % fHist(rem(frameNum,saveDataFrames)).numPeople = true_num_people;
    %         fHist(rem(frameNum,saveDataFrames)).sensorPosition = Params.sensorPosition;
    %         fHist(rem(frameNum,saveDataFrames)).dt = dt;
    %         fHist(rem(frameNum,saveDataFrames)).unix_time = unix_time;
    %         fHist(rem(frameNum,saveDataFrames)).num_people_frame = num_people_frame;
    %     end
    % end


    % Get the size of the 2D array
    [m, n] = size(rangeAngle);
    
    if playBack == 0
        % Get the current size of the 3D array
        [~, ~, p] = size(heatmapsToSave);
    
        % Check if the 3D array is empty
        if isempty(heatmapsToSave)
            % If the 3D array is empty, initialize it with the 2D array
            heatmapsToSave = zeros(m, n, 1);
            heatmapsToSave(:, :, 1) = rangeAngle;
        else
            % If the 3D array is not empty, append the 2D array to it
            heatmapsToSave(:, :, p + 1) = rangeAngle;
        end
    end
    if isempty(rangeAngle)
        continue
    end
    combinedHeatmaps(:,:,mod(frameNum-1, numFramesToAccumulate) + 1) = rangeAngle;

    if heatmapCollapseMethod == 1
        collapsedHeatmap = mean(combinedHeatmaps, 3);
    elseif heatmapCollapseMethod == 2
        collapsedHeatmap = max(combinedHeatmaps, [], 3);
    end

    num_people_frame = 0;
    fheatmap = collapsedHeatmap(1:64,:);
    rejectedPixels = [];
    load('rangeBasedThresholds.mat');
    y_fit(1:41) = 200;
    y_fit = y_fit*sensitivity.Value;
    otherValidClusters = {};
    eligibleValidClusters = {};
    
    oneThirdBin = round(maxRange/Params.dataPath.rangeResolutionMeters/3);
    finalImage = zeros(size(collapsedHeatmap));
    finalImage(1:oneThirdBin,:) = watershedByParts(collapsedHeatmap(1:oneThirdBin,:),collapsedHeatmap,hmin1,otsuMultiplier);
%     imshow(finalImage); 
    finalImage(oneThirdBin+1:2*oneThirdBin,:) = watershedByParts(collapsedHeatmap(oneThirdBin+1:2*oneThirdBin,:),collapsedHeatmap,hmin2,otsuMultiplier);
%     imshow(finalImage);
    finalImage(2*oneThirdBin+1:end,:) = watershedByParts(collapsedHeatmap(2*oneThirdBin+1:end,:),collapsedHeatmap,hmin2,otsuMultiplier);
%     imshow(finalImage);

    % Find the connected components in the thresholded segment
    CC = bwconncomp(~finalImage);

    % Store each connected component and its corresponding Otsu threshold level in the structure array
    for j = 1:CC.NumObjects
        maxValue = max(fheatmap(CC.PixelIdxList{j}));
        [row, col] = ind2sub(size(fheatmap), find(fheatmap == maxValue, 1));
        rangeBlob = Params.dataPath.rangeResolutionMeters*row;
        % Calculate the depth of the catchment basin
        if maxValue > y_fit(row) && size(CC.PixelIdxList{j},1) >= minPoints && rangeBlob <= maxRange
%             if size(CC.PixelIdxList{j},1) >= minPoints && rangeBlob <= maxRange
            num_people_frame = num_people_frame + 1;
%                 if row < rMaxCombination
                eligibleValidClusters{end+1} = CC.PixelIdxList{j};
%                 else
%                     otherValidClusters{end+1} = CC.PixelIdxList{j};
%                 end
        else
            rejectedPixels = [rejectedPixels;CC.PixelIdxList{j}];   
        end     
    end

%     minVal = 0;
%     maxVal = max(collapsedHeatmap(:));
%     
%     % Normalize the matrix to [0, 1]
%     normalizedMatrix = (collapsedHeatmap - minVal) / (maxVal - minVal);
%     ms = 1 - normalizedMatrix(1:64,:);
%     ms = uint8(ms * 255);
% 
%     hs = watershed(imhmin(ms,hmin));         
%     num_segments = cast(max(hs(:)),'int8');
% 
%     final_segmanted = zeros(size(ms));   
%     % Loop over each segment
%     for k = 1:num_segments
%         % Create a binary mask for the current segment
%         mask = hs == k;
%         
%         % Apply the mask to the image
%         segment = ms .* uint8(mask);
%         
%         % Apply Otsu's thresholding to the segment
%         level = graythresh(segment(mask));
%         bw = imbinarize(segment, level*otsuMultiplier);
% 
%         % Store the thresholded segment in the cell array
%         final_segmanted = final_segmanted | bw;
% 
%         % Find the connected components in the thresholded segment
%         CC = bwconncomp(~(bw | ~mask));
% 
%         % Store each connected component and its corresponding Otsu threshold level in the structure array
%         for j = 1:CC.NumObjects
%             maxValue = max(fheatmap(CC.PixelIdxList{j}));
%             [row, col] = ind2sub(size(fheatmap), find(fheatmap == maxValue, 1));
%             rangeBlob = Params.dataPath.rangeResolutionMeters*row;
%             % Calculate the depth of the catchment basin
%             if maxValue > y_fit(row) && size(CC.PixelIdxList{j},1) >= minPoints && rangeBlob <= maxRange
% %             if size(CC.PixelIdxList{j},1) >= minPoints && rangeBlob <= maxRange
%                 num_people_frame = num_people_frame + 1;
% %                 if row < rMaxCombination
%                     eligibleValidClusters{end+1} = CC.PixelIdxList{j};
% %                 else
% %                     otherValidClusters{end+1} = CC.PixelIdxList{j};
% %                 end
%             else
%                 rejectedPixels = [rejectedPixels;CC.PixelIdxList{j}];   
%             end     
%         end
%     end
%     finalImage = hs==0 | final_segmanted;

%     adjMat = combineClusters(eligibleValidClusters, size(hs),hs==0 | final_segmanted);
%     adjMat = combineClustersAggressive(eligibleValidClusters, size(hs),hs==0 | final_segmanted, 5);
    adjMat = combineClustersAggressiveRect(eligibleValidClusters, size(finalImage),finalImage, 9, 3);

%     if num_people_frame <= 3 && sum(collapsedHeatmap,"all") > t4
%         num_people_frame = num_people_frame + 1;
%     elseif num_people_frame <= 4 && sum(collapsedHeatmap,"all") > t5
%         num_people_frame = num_people_frame + 1;
%     end

    img = plotClusters(eligibleValidClusters, otherValidClusters, adjMat, size(finalImage));
    CC = bwconncomp(~img);
    num_people_frame = CC.NumObjects;

    peopleCountBuffer(2:peopleCountingBufferSize) = peopleCountBuffer(1:peopleCountingBufferSize-1);
    if num_people_frame >= 5
        num_people_frame = 5;
    end
    peopleCountBuffer(1) = num_people_frame;

    if saveBool == 1
        if rem(frameNum,saveDataFrames) == 0
            % Check if the folder exists, if not, create it
            if ~exist(folderPath, 'dir')
                mkdir(folderPath);
            end

            % Get a list of .mat files in the folder
            files = dir(fullfile(folderPath, '*.mat'));

            % Get the number of existing files
            numFiles = length(files);

            % Find the largest file number in the folder
            maxFileNum = 0;
            for i = 1:numFiles
                % Extract the file number from the filename
                [~, name, ~] = fileparts(files(i).name);
                fileNum = str2double(regexp(name, '\d*', 'match'));

                % Update the maximum file number
                if fileNum > maxFileNum
                    maxFileNum = fileNum;
                end
            end

            % Define the next integer y
            y = maxFileNum + 1;

            % Define the filename
            filename = sprintf('%d.mat', y);

            % Full file path
            fullFilePath = fullfile(folderPath, filename);

            % Save 'fHist' to a .mat file
            save(fullFilePath, 'fHist');
            disp(fprintf('Saved %s',fullFilePath));

            fHist = struct;
        else
            fHist(rem(frameNum,saveDataFrames)).rangeAzimuthHeatMapMinor = rangeAngle;
            fHist(rem(frameNum,saveDataFrames)).room = room;
            % fHist(rem(frameNum,saveDataFrames)).numPeople = true_num_people;
            fHist(rem(frameNum,saveDataFrames)).sensorPosition = Params.sensorPosition;
            fHist(rem(frameNum,saveDataFrames)).dt = dt;
            fHist(rem(frameNum,saveDataFrames)).unix_time = unix_time;
            fHist(rem(frameNum,saveDataFrames)).num_people_frame = num_people_frame;
        end
    end
    
    accuracy_frame = 100 - 100*abs(mode(peopleCountBuffer(peopleCountBuffer >= 0)) - true_num_people)/true_num_people;
    accuracy_frame_mean = 100 - 100*abs(mean(peopleCountBuffer(peopleCountBuffer >= 0)) - true_num_people)/true_num_people;
    if mode(peopleCountBuffer(peopleCountBuffer >= 0)) <= true_num_people
        accuraccyGreaterThan_frame = 100;
    else
        accuraccyGreaterThan_frame = 100 - 100*abs(mode(peopleCountBuffer(peopleCountBuffer >= 0)) - true_num_people)/true_num_people;
    end
    accuracy = ((frameNum - numFramesToAccumulate - 1)*accuracy + accuracy_frame)/(frameNum - numFramesToAccumulate);
    accuracy_mean = ((frameNum - numFramesToAccumulate - 1)*accuracy_mean + accuracy_frame_mean)/(frameNum - numFramesToAccumulate);
    accuraccyGreaterThan = ((frameNum - numFramesToAccumulate - 1)*accuraccyGreaterThan + accuraccyGreaterThan_frame)/(frameNum - numFramesToAccumulate);
    if frameNum <= numFramesToAccumulate
        accuracy = 0;
        accuraccyGreaterThan = 0;
        accuracy_mean = 0;
    end

    im2show = finalImage;
    % Convert the image to double data type for processing
    im2show = im2double(im2show);
    
    % Set the pixels at the specified indices to gray (0.5 for normalized images)
    im2show(rejectedPixels) = 0.5;
    
%     collapsedHeatmap(round(maxRange/Params.dataPath.rangeResolutionMeters),:) = 1000;
    matrixSize = size(collapsedHeatmap);  % Get the size of your matrix
    linearIndices = cellfun(@(point) sub2ind(matrixSize, point(1), point(2)), boundaryPoints);
%     collapsedHeatmap(linearIndices) = 1000;
%     collapsedHeatmap(rMaxCombination, :) = 500; 
    
    if rem(frameNum,2)==0 && ~exist('outsideRun','var')
        subplot(1,3,1)
        s = surf(collapsedHeatmap);view(2);
        s.YData = 0:Params.dataPath.rangeResolutionMeters:Params.dataPath.rangeResolutionMeters*(Params.chirpComnCfg.numAdcSamples/2 - 1);
        w = [-pi:2*pi/FFTsize_Angle_Az:pi-2*pi/FFTsize_Angle_Az];                                 % The sampled FFT Frequencies                        
        angles = asind(w/pi);
        s.XData = angles;
        xlabel('Angle (degree)')
        ylabel('Range (m)')
        title("Range-Azimuth Heatmap")
%         subplot(2,2,4)
%         s = surf(20*log10(collapsedHeatmap));view(2);
%         s.YData = 0:Params.dataPath.rangeResolutionMeters:Params.dataPath.rangeResolutionMeters*(Params.chirpComnCfg.numAdcSamples/2 - 1);
%         w = [-pi:2*pi/FFTsize_Angle_Az:pi-2*pi/FFTsize_Angle_Az];                                 % The sampled FFT Frequencies                        
%         angles = asind(w/pi);
%         s.XData = angles;
%         xlabel('Angle (degree)')
%         ylabel('Range (m)')
%         title("Range-Azimuth Heatmap (log)")
        subplot(1,3,2)
        imshow(flip(im2show));
        title("Intermediate Image (for debugging)")
        subplot(1,3,3)
        imshow(flip(img));
        title("Final image to count people")
        sgtitle(sprintf('Frame No.: %d\nNumber of People (Frame) = %d\nNumber of People (Mode) = %d\nAccuracy = %f\nAccuracy (mean) = %f\n(Note: Any count >= 5 is displayed as 5)', frameNum,num_people_frame, mode(peopleCountBuffer(peopleCountBuffer >= 0)), accuracy, accuracy_mean));
        if playBack == 1
            pause(0.001)
        end
    end
%%
    if playBack == 1 && frameNum == size(fHistfile.fHist,2)
        break
    end

%% Ending Formalities 
    % if playBack == 0
    %     lostSyncTime = tic;
    %     bytesAvailable = hDataSerialPort.NumBytesAvailable;
    %     disp(['Lost sync at frame ', num2str(targetFrameNum),'(', num2str(frameNum), '), Reason: ', reason, ', ', num2str(bytesAvailable), ' bytes in Rx buffer']);
    % 
    %     outOfSyncBytes = 0;
    %     while(lostSync)
    %         syncPatternFound = 1;
    %         for n=1:8
    %             [rxByte, byteCount] = readSerial(hDataSerialPort, 1, 'uint8');
    %             if(rxByte ~= syncPatternUINT8(n))
    %                 syncPatternFound = 0;
    %                 outOfSyncBytes = outOfSyncBytes + 1;
    %                 break;
    %             end
    %         end
    %         if(syncPatternFound == 1)
    %             lostSync = 0;
    %             frameNum = frameNum + 1;
    % 
    %             [header, byteCount] = readSerial(hDataSerialPort, frameHeaderLengthInBytes - 8, 'uint8');
    %             rxHeader = [syncPatternUINT8, header];
    %             byteCount = byteCount + 8;
    %             gotHeader = 1;
    %         end
    %     end
    % else
    %     break
    % end
  end
end
   
%%
if exist('outsideRun','var')
    if ~outsideRun
        fprintf("Accuracy = %f\n\n",accuracy);
        disp('Done');
    end
end

function rangeAngle = doa_rangeAngleHeatmapGen_capon(rangeProfile, doaParams)
    rangeProfile = permute(rangeProfile,[2 3 1]);

    % calibration
    rangeProfile = rangeProfile .* repmat(doaParams.phaseCompArray, 1, size(rangeProfile,2), size(rangeProfile,3));
    
    % prepare data
    [~, ~, M]           = size(rangeProfile);
    steerMat            = doaParams.steerMat;
    nTheta              = size(steerMat, 2);

    % Some other parameters needed for some options
    uniqueAzimRows  = unique(doaParams.m_ind);
    azimRowRange    = max(uniqueAzimRows) - min(uniqueAzimRows) + 1;
    uniqueElevRows  = unique(doaParams.n_ind);
    elevRowRange    = max(uniqueElevRows) - min(uniqueElevRows) + 1;

    m_ind = abs(doaParams.m_ind) + 1;
    n_ind = abs(doaParams.n_ind) + 1;
    m_ind = m_ind - min(m_ind) + 1; % keep the first index as 1
    n_ind = n_ind - min(n_ind) + 1; % keep the first index as 1

    fftDim = 1;
    nAnt  = length(doaParams.virtualAntToProcess);
    
    rangeAngle          = zeros(M, nTheta/2,1);
    condNumber          = zeros(M, fftDim);
    bfWeight            = zeros(M, nAnt, nTheta, fftDim); 
    invCovMat           = zeros(M, nAnt, nAnt, fftDim);

    % compute the range-angle heatmap
    for rIdx = 1:M  % rIdx: range index
        % extract the signal: use the first rx.doa.cov_numChirps chirps
        sigMat              = squeeze(rangeProfile(:, :, rIdx));
        
        % processing in Doppler domain
        fftSize_Doppler     = doaParams.FFTsize_Doppler*doaParams.numFrmPerMinorMotProc;
        
        means = mean(sigMat,2);
        sigMat = sigMat - means;
        
        sigMat = sigMat([1 4 3 6],:);
        
    
        % Process every elevation bin (if any)
        for eIdx = 1:size(sigMat,3)
            % estimate the covariance matrix
            covMat = (sigMat(:,:,eIdx) * sigMat(:,:,eIdx)') / size(sigMat,2);
            
            % for debugging
            condNumber(rIdx,eIdx)    = cond(covMat);
    
            % diagonally loading to improve stability
            alpha               = 0.03 * mean(diag(covMat));
            covMat              = covMat + alpha*eye(size(covMat));
    
            % inverse of the covariance
            pinv_covMat         = pinv(covMat);
            
            w = [-pi:2*pi/doaParams.FFTsize_Angle_Az:pi-2*pi/doaParams.FFTsize_Angle_Az];                                 % The sampled FFT Frequencies                        
            angles = asind(w/pi);
            caponSpectrum = zeros(1,doaParams.FFTsize_Angle_Az/2);
            for i = 1:doaParams.FFTsize_Angle_Az
                s=exp(1i*pi*sind(angles(i))*[0:3]); %steering vector
                s=s';
                s1=s(1:4);
                caponSpectrum(1,i)=abs(1/(s1'*pinv_covMat*s1));
            end
            rangeAngle(rIdx,:) = caponSpectrum;
        end
    end
end

function data = saturateData(data, numDataBits, numMaxbits)
    % Saturate the data
    %
    % Created       : April, 2022 by Muhammet Emin Yanik (based on Slobodan's inputs)
    % Last Modified : April, 2022 by Muhammet Emin Yanik
    
    % Compute the maximum mand minimum limits
    numSaturationBits = min(numMaxbits,numDataBits)-1;
    maxLimit = 2^(numSaturationBits)-1;
    minLimit = -1 * 2^(numSaturationBits);
    
    % Saturate the data
    if isreal(data)
        data = min(data, maxLimit);
        data = max(data, minLimit);
    else
        dataReal = min(real(data), maxLimit);
        dataReal = max(dataReal, minLimit);
    
        dataImag = min(imag(data), maxLimit);
        dataImag = max(dataImag, minLimit);
    
        data = complex(dataReal,dataImag);
    end
end

% Read the frame header
function [rxHeader, byteCount, outOfSyncBytes] = readFrameHeader(hDataSerialPort, frameHeaderLengthInBytes, syncPatternUINT8)
    lostSync = 1;
    outOfSyncBytes = 0;
    while(lostSync)
        syncPatternFound = 1;
        for n=1:8          
            [rxByte, byteCount] = readSerial(hDataSerialPort, 1, 'uint8');
            if(rxByte ~= syncPatternUINT8(n))
                syncPatternFound = 0;
                outOfSyncBytes = outOfSyncBytes + 1;
                break;
            end
        end
        if(syncPatternFound == 1)
            lostSync = 0;            
            [header, byteCount] = readSerial(hDataSerialPort, frameHeaderLengthInBytes - 8, 'uint8');
            rxHeader = [syncPatternUINT8, header];
            byteCount = byteCount + 8;
        end
    end
end

% A wrapper function to read data from serial
function [dataRead, countRead] = readSerial(device, count, datatype)
    dataRead = read(device, count, datatype);
    dataRead = cast(dataRead,datatype);
    countRead = length(dataRead);
end

function [P] = parseCfg6432(cliCfg)
    P=[];
    P.adcDataSource.isSourceFromFile = 0;
    P.adcDataSource.numFrames = 0;
    numBoxes = 0;
    
    for k=1:length(cliCfg)
        C = strsplit(cliCfg{k});
        % Sensor Front-End Parameters
        if strcmp(C{1},'channelCfg')
            P.channelCfg.txChannelEn = str2double(C{3});
            P.dataPath.numTxAnt = bitand(bitshift(P.channelCfg.txChannelEn, 0),1) +...
                                      bitand(bitshift(P.channelCfg.txChannelEn,-1),1);
            P.channelCfg.rxChannelEn = str2double(C{2});
            P.dataPath.numRxAnt = bitand(bitshift(P.channelCfg.rxChannelEn,0),1) +...
                                  bitand(bitshift(P.channelCfg.rxChannelEn,-1),1) +...
                                  bitand(bitshift(P.channelCfg.rxChannelEn,-2),1);                         

        elseif strcmp(C{1},'chirpComnCfg')
            P.chirpComnCfg.digOutputSampRate  = 100 / str2double(C{2}); % in MHz
            P.chirpComnCfg.digOutputBitsSel   = str2double(C{3});
            P.chirpComnCfg.dfeFirSel          = str2double(C{4});
            P.chirpComnCfg.numAdcSamples      = str2double(C{5});
            P.chirpComnCfg.chirpTxMimoPatSel  = str2double(C{6});
            P.chirpComnCfg.chirpRampEndTime   = str2double(C{7}); % in us
            P.chirpComnCfg.chirpRxHpfSel      = str2double(C{8});

        elseif strcmp(C{1},'chirpTimingCfg')
            P.chirpTimingCfg.idleTime = str2double(C{2}); %in us
            P.chirpTimingCfg.numSkippedSamples = str2double(C{3}); %in adc samples
            P.chirpTimingCfg.ChirpTxStartTime = str2double(C{4}); %in us
            P.chirpTimingCfg.chirpSlope = str2double(C{5}); % in MHz/us
            P.chirpTimingCfg.startFreq = str2double(C{6});  % in GHz

        elseif strcmp(C{1},'frameCfg')
            P.frameCfg.numOfChirpsInBurst = str2double(C{2});
            P.frameCfg.numOfChirpsAccum = str2double(C{3});
            P.frameCfg.burstPeriodicity = str2double(C{4});
            P.frameCfg.numOfBurstsInFrame = str2double(C{5});
            P.frameCfg.framePeriodicity = str2double(C{6});
            P.frameCfg.numFrames = str2double(C{7});
            P.frameCfg.numLoops = P.frameCfg.numOfChirpsInBurst * P.frameCfg.numOfBurstsInFrame / P.dataPath.numTxAnt;
            P.frameCfg.chirpStartIdx = 0;
            P.frameCfg.chirpEndIdx = P.dataPath.numTxAnt - 1;

        
        elseif strcmp(C{1},'factoryCalibCfg')
            P.factoryCalibCfg.saveEnable = str2double(C{2});
            P.factoryCalibCfg.restoreEnable = str2double(C{3});
            P.factoryCalibCfg.rxGain = str2double(C{4});
            P.factoryCalibCfg.backoff = str2double(C{5});
            P.factoryCalibCfg.flashOffset = dec2hex(str2double(C{6}));
            
        % Detection Layer Parameters
        elseif strcmp(C{1},'sigProcChainCfg')
            P.dataPath.azimuthFftSize = str2double(C{2});
            P.dataPath.elevationFftSize = str2double(C{3});
            P.sigProcChainCfg.motDetMode = str2double(C{4});
        
        elseif strcmp(C{1},'guiMonitor')
            P.guiMonitor.pointCloud = str2double(C{2});
            P.guiMonitor.rangeProfileMask = str2double(C{3});
            P.guiMonitor.rangeProfileMajor = bitand(P.guiMonitor.rangeProfileMask,1);
            P.guiMonitor.rangeProfileMinor = bitshift(bitand(P.guiMonitor.rangeProfileMask, 2), -1);
            P.guiMonitor.heatMapMask = str2double(C{5});
            P.guiMonitor.heatMapMajor = bitand(P.guiMonitor.heatMapMask,1);
            P.guiMonitor.heatMapMinor = bitshift(bitand(P.guiMonitor.heatMapMask, 2), -1);
            P.guiMonitor.statsInfo = str2double(C{7});
            if (length(C) >= 8), P.guiMonitor.presenceDetected = str2double(C{8}); end
            if (length(C) >= 9), P.guiMonitor.adcSamples = str2double(C{9}); end
            if (length(C) >= 10), P.guiMonitor.trackerInfo = str2double(C{10}); end
            if (length(C) >= 11), P.guiMonitor.microDopplerInfo = str2double(C{11}); end
            if (length(C) >= 12), P.guiMonitor.classifierInfo = str2double(C{12}); end
            
        elseif strcmp(C{1},'adcDataSource')
            if str2double(C{2}) == 1
            	P.adcDataSource.isSourceFromFile = 1;
                [P.adcDataSource.adcDataFilePath, P.adcDataSource.adcDataFileName, adcDataFileNameExt] = fileparts(C{3});
                P.adcDataSource.adcDataFileName = [P.adcDataSource.adcDataFileName adcDataFileNameExt];
                [adcFid, errmsg] = fopen([P.adcDataSource.adcDataFilePath, '/', P.adcDataSource.adcDataFileName], 'rb');
                if ~isempty(errmsg)
                    error(errmsg);
                end
                P.adcDataSource.numAdcSample = fread(adcFid, 1, 'int32');
                P.adcDataSource.numVirtualAntennas = fread(adcFid, 1, 'int32');
                P.adcDataSource.numChirpsPerFrame = fread(adcFid, 1, 'int32');
                P.adcDataSource.numFrames = fread(adcFid, 1, 'int32');
                fclose(adcFid);
            else
                P.adcDataSource.isSourceFromFile = 0;
                P.adcDataSource.numFrames = 0;
            end
        
        % High-Level Processing Layer Parameters
        elseif strcmp(C{1},'mpdBoundaryBox')
            numBoxes = numBoxes + 1;
            P.mpdBoundaryBox.numBoxes = numBoxes;
            assert(P.mpdBoundaryBox.numBoxes == str2double(C{2}), 'Configure the boundary boxes sequentially!');
            P.mpdBoundaryBox.idx(P.mpdBoundaryBox.numBoxes)  = str2double(C{2});
            P.mpdBoundaryBox.xMin(P.mpdBoundaryBox.numBoxes) = str2double(C{3});
            P.mpdBoundaryBox.xMax(P.mpdBoundaryBox.numBoxes) = str2double(C{4});
            P.mpdBoundaryBox.yMin(P.mpdBoundaryBox.numBoxes) = str2double(C{5});
            P.mpdBoundaryBox.yMax(P.mpdBoundaryBox.numBoxes) = str2double(C{6});
            P.mpdBoundaryBox.zMin(P.mpdBoundaryBox.numBoxes) = str2double(C{7});
            P.mpdBoundaryBox.zMax(P.mpdBoundaryBox.numBoxes) = str2double(C{8}); 
        
        elseif strcmp(C{1},'sensorPosition')
            P.sensorPosition.xOffset = str2double(C{2});
            P.sensorPosition.yOffset = str2double(C{3});
            P.sensorPosition.zOffset = str2double(C{4});
            P.sensorPosition.azimuthTilt = str2double(C{5});
            P.sensorPosition.elevationTilt = str2double(C{6});
        
        elseif strcmp(C{1},'clusterCfg')
            P.clusterCfg.enabled = str2double(C{2});
            P.clusterCfg.maxDistance = str2double(C{3});
            P.clusterCfg.minPoints = str2double(C{4});
        
        elseif strcmp(C{1},'majorStateCfg')
            P.majorStateCfg.pointThre1 = str2double(C{2});
            P.majorStateCfg.pointThre2 = str2double(C{3});
            P.majorStateCfg.snrThre2 = str2double(C{4});
            P.majorStateCfg.pointHistThre1 = str2double(C{5});
            P.majorStateCfg.pointHistThre2 = str2double(C{6});
            P.majorStateCfg.snrHistThre2 = str2double(C{7});
            P.majorStateCfg.histBufferSize = str2double(C{8});
            P.majorStateCfg.major2minorThre = str2double(C{9});
        
        elseif strcmp(C{1},'minorStateCfg')
            P.minorStateCfg.pointThre1 = str2double(C{2});
            P.minorStateCfg.pointThre2 = str2double(C{3});
            P.minorStateCfg.snrThre2 = str2double(C{4});
            P.minorStateCfg.pointHistThre1 = str2double(C{5});
            P.minorStateCfg.pointHistThre2 = str2double(C{6});
            P.minorStateCfg.snrHistThre2 = str2double(C{7});
            P.minorStateCfg.histBufferSize = str2double(C{8});
            P.minorStateCfg.minor2emptyThre = str2double(C{9});
        
        % Tracker Processing Layer Parameters
        elseif strcmp(C{1},'trackingCfg')
            P.trackingCfg = str2double(C(2:8));

        elseif strcmp(C{1},'boundaryBox')
            P.boundaryBox = str2double(C(2:7));
        
        elseif strcmp(C{1},'staticBoundaryBox')
            P.staticBoundaryBox = str2double(C(2:7));
        
        elseif strcmp(C{1},'presenceBoundaryBox')
            P.presenceBoundaryBox = str2double(C(2:7));

        elseif strcmp(C{1},'gatingParam')
            P.gatingParam = str2double(C(2:6));
        
        elseif strcmp(C{1},'stateParam')
            P.stateParam = str2double(C(2:7));

        elseif strcmp(C{1},'allocationParam')
            P.allocationParam = str2double(C(2:7));

        elseif strcmp(C{1},'maxAcceleration')
            P.maxAcceleration = str2double(C(2:4));

        % Micro-Doppler Processing Layer Parameters
        elseif strcmp(C{1},'microDopplerCfg')
            P.microDopplerCfg.enabled = str2double(C{2});

        % Classification Layer Parameters
        elseif strcmp(C{1},'classifierCfg')
            P.classifierCfg.enabled = str2double(C{2});
            P.classifierCfg.minNpntsPerTrack = str2double(C{3});
            P.classifierCfg.missTotFrmThre = str2double(C{4});
        
        % Vital Signs Layer Parameters
        elseif strcmp(C{1},'vitalsPCCfg')
            P.vitalsPCCfg.rangebinIdx = str2double(C{2});
            P.vitalsPCCfg.numRangeBins = str2double(C{3});
            P.vitalsPCCfg.vitalsPCFlag = str2double(C{4});
       
        % Rx channel compensation
        elseif strcmp(C{1},'measureRangeBiasAndRxChanPhase')
            P.measureRangeBiasAndRxChanPhase.enabled = str2double(C{2});
        
        end
    end

    % Set profileCfg
    P.profileCfg.startFreq =  P.chirpTimingCfg.startFreq;
    if P.frameCfg.numOfBurstsInFrame > 1
        P.profileCfg.idleTime(1) =  P.frameCfg.burstPeriodicity - ...
            P.frameCfg.numOfChirpsInBurst*(P.chirpTimingCfg.idleTime + P.chirpComnCfg.chirpRampEndTime) + ...
            P.chirpTimingCfg.idleTime;
    else
        P.profileCfg.idleTime(1) =  P.chirpTimingCfg.idleTime;
    end
    P.profileCfg.idleTime(2) =  P.chirpTimingCfg.idleTime;
    P.profileCfg.rampEndTime = P.chirpComnCfg.chirpRampEndTime;
    P.profileCfg.freqSlopeConst = P.chirpTimingCfg.chirpSlope;
    P.profileCfg.numAdcSamples = P.chirpComnCfg.numAdcSamples;
    P.profileCfg.digOutSampleRate = 1000 * P.chirpComnCfg.digOutputSampRate;
    
    % Set motion mode flags
    P.sigProcChainCfg.enableMajorMotion = bitand(P.sigProcChainCfg.motDetMode, 1);
    P.sigProcChainCfg.enableMinorMotion = bitshift(bitand(P.sigProcChainCfg.motDetMode, 2), -1);

    % Set the other data path parameters
    P.dataPath.numChirpsPerFrame = (P.frameCfg.chirpEndIdx -...
                                            P.frameCfg.chirpStartIdx + 1) *...
                                            P.frameCfg.numLoops;
    P.dataPath.numDopplerChirps = P.dataPath.numChirpsPerFrame / P.dataPath.numTxAnt;
    P.dataPath.numDopplerBins = 2^ceil(log2(P.dataPath.numDopplerChirps));
    P.dataPath.numRangeBins = pow2roundup(P.profileCfg.numAdcSamples);
    P.dataPath.numValidRangeBins = P.dataPath.numRangeBins/2;  %Real ADC samples
    
    P.dataPath.rangeResolutionMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
                     (2 * abs(P.profileCfg.freqSlopeConst) * 1e12 * P.profileCfg.numAdcSamples);

    if P.profileCfg.startFreq >= 76
        CLI_FREQ_SCALE_FACTOR =(3.6);  %77GHz
    else
        CLI_FREQ_SCALE_FACTOR =(2.7); %60GHz
    end
    mmwFreqSlopeConst = fix(P.profileCfg.freqSlopeConst * (2^26) /((CLI_FREQ_SCALE_FACTOR * 1e3) * 900.0));
    P.dataPath.rangeIdxToMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
                         (2 * abs(mmwFreqSlopeConst) *  ((CLI_FREQ_SCALE_FACTOR*1e3*900)/(2^26))* 1e12 * P.dataPath.numRangeBins);
                     
    startFreqConst = fix(P.profileCfg.startFreq * (2^26) /CLI_FREQ_SCALE_FACTOR);
    P.dataPath.dopplerResolutionMps = 3e8 /...
        (2 * startFreqConst / 67108864*CLI_FREQ_SCALE_FACTOR*1e9 * ...
         ((P.profileCfg.idleTime(1) + P.profileCfg.rampEndTime)+ ...
          (P.profileCfg.idleTime(2) + P.profileCfg.rampEndTime))*1e-6 * ...
         P.dataPath.numDopplerBins);                                    
end

function sphandle = configureControlPort(comPortNum,baudrate)
    comPortsAvailable = serialportlist("available");
    comPortString = ['COM' num2str(comPortNum)];
    if any(contains(comPortsAvailable,comPortString))
        sphandle = serialport(comPortString,baudrate,'Parity','none','Timeout',10); 
        configureTerminator(sphandle,'CR/LF');
        flush(sphandle);
    else
        sphandle = [];
        fprintf('Serial port %s is already open or not available! Power cycle the device and re-run the application...\n', comPortString);
    end
end

function sphandle = reconfigureControlPort(sphandle)
    configureTerminator(sphandle,0);
end

function writelineslow(sphandle, cliCfg)
    configureTerminator(sphandle,0);
    for n = 1:length(cliCfg)
        if n~=length(cliCfg)
            write(sphandle,cliCfg(n),"char")
            pause(0.002);
        else
            configureTerminator(sphandle,'CR/LF');
            writeline(sphandle,cliCfg(n));
        end
    end
end

function config = readCfg(filename)
    config = cell(1,100);
    fid = fopen(filename, 'r');
    if fid == -1
        fprintf('File %s not found!\n', filename);
        return;
    else
        fprintf('Opening configuration file %s ...\n', filename);
    end
    tline = fgetl(fid);
    k=1;
    while ischar(tline)
        config{k} = tline;
        tline = fgetl(fid);
        k = k + 1;
    end
    config = config(1:k-1);
    fclose(fid);
end

function length = lengthFromStruct(S)
    fieldName = fieldnames(S);
    length = 0;
    for n = 1:numel(fieldName)
        [~, fieldLength] = S.(fieldName{n});
        length = length + fieldLength;
    end
end

function [R] = readToStruct(S, ByteArray)
    fieldName = fieldnames(S);
    offset = 0;
    for n = 1:numel(fieldName)
        [fieldType, fieldLength] = S.(fieldName{n});
        R.(fieldName{n}) = typecast(uint8(ByteArray(offset+1:offset+fieldLength)), fieldType);
        offset = offset + fieldLength;
    end
end

function [y] = pow2roundup (x)
    y = 1;
    while x > y
        y = y * 2;
    end
end

function [fig, sensitivity] = createSliderFigure(outsideRun)    
    % Create a Sensitivity object
    sensitivity = Sensitivity;

    if exist('outsideRun','var')
        if(~outsideRun)
            % Create a figure
            fig = figure;
        
            % Define the slider's width and height (as a fraction of the figure's width and height)
            sldWidth = 0.5;  % 50% of the figure's width
            sldHeight = 0.05;  % 5% of the figure's height
        
            % Calculate the left and bottom positions for the slider (as a fraction of the figure's width and height)
            sldLeft = 1 - sldWidth;  % aligns the slider to the right
            sldBottom = 0.1;  % 10% from the bottom
        
            % Create a slider in the figure
            sld = uicontrol('Parent', fig, 'Style', 'slider', 'Units', 'normalized', 'Position', [sldLeft, sldBottom, sldWidth, sldHeight], 'BackgroundColor', [0.8 0.8 0.8]);
        
            % Set the slider's min, max, and step values
            sld.Min = 0;
            sld.Max = 1;
            sld.Value = sensitivity.Value;  % set the initial slider value
            sld.SliderStep = [0.05, 0.05];  % major and minor steps of 0.05
        
            % Create a label for the slider
            lbl = uicontrol('Parent', fig, 'Style', 'text', 'Units', 'normalized', 'Position', [sldLeft, sldBottom + sldHeight, sldWidth, 0.05], 'String', ['Count Sensitivity = ', num2str(sensitivity.Value)], 'BackgroundColor', [0.8 0.8 0.8]);
        
            % Create a text box to display the current slider value
            txt = uicontrol('Parent', fig, 'Style', 'text', 'Units', 'normalized', 'Position', [sldLeft + sldWidth, sldBottom, 0.1, 0.05], 'BackgroundColor', [0.8 0.8 0.8]);
        
            % Set the slider's callback function to update the text box, the label, and the sensitivity handle
            sld.Callback = @(src, ~) updateSensitivity(src, lbl, txt, sensitivity);
        end

        fig = 0;
    end
    
    
end

function updateSensitivity(src, lbl, txt, sensitivity)
    % Update the text box and the sensitivity handle
    set(txt, 'String', num2str(src.Value));
    sensitivity.Value = src.Value;

    % Update the label text
    set(lbl, 'String', ['Sensitivity = ', num2str(src.Value)]);
end

function [heatmap,boundaryPoints] = getBoundaryPoints(FFTsize_Angle_Az, Params)
    rotationMatrix = [cosd(Params.sensorPosition.azimuthTilt) sind(Params.sensorPosition.azimuthTilt); -sind(Params.sensorPosition.azimuthTilt) cosd(Params.sensorPosition.azimuthTilt)];
    translationMatrix = [Params.sensorPosition.xOffset; Params.sensorPosition.yOffset];
    heatmap = zeros([Params.chirpComnCfg.numAdcSamples/2,FFTsize_Angle_Az]);
    w = [-pi:2*pi/FFTsize_Angle_Az:pi-2*pi/FFTsize_Angle_Az];                                 % The sampled FFT Frequencies                        
    angles = asind(w/pi);
    boundaryPoints = {};

    for i=1:Params.chirpComnCfg.numAdcSamples/2
        for j=1:FFTsize_Angle_Az
            theta = angles(j);
            r = i*Params.dataPath.rangeResolutionMeters;
            point = rotationMatrix*[r*sind(theta); r*cosd(theta)] + translationMatrix;
            if point(1) >= Params.mpdBoundaryBox.xMin && point(1) <= Params.mpdBoundaryBox.xMax && point(2) >= Params.mpdBoundaryBox.yMin && point(2) <= Params.mpdBoundaryBox.yMax
                heatmap(i,j) = 1000;
            end
            
            if j >= 2
                if heatmap(i,j) - heatmap(i,j-1) == 1000
                    boundaryPoints{end+1} = [i,j-1];
                elseif heatmap(i,j) - heatmap(i,j-1) == -1000
                    boundaryPoints{end+1} = [i,j];
                end
            end
        end
    end

    for j=1:FFTsize_Angle_Az
        for i=2:Params.chirpComnCfg.numAdcSamples/2
            if heatmap(i,j) - heatmap(i-1,j) == 1000
                boundaryPoints{end+1} = [i-1,j];
            elseif heatmap(i,j) - heatmap(i-1,j) == -1000
                boundaryPoints{end+1} = [i,j];
            end
        end
    end
end

function params = parseFile(filename)
    % Check if the file exists
    if ~isfile(filename)
        params = {};
        disp('File not found')
        return;
    end
    
    % Open the file
    fid = fopen(filename, 'r');
    
    % Initialize an empty cell array
    params = {};
    
    % Read the file line by line
    tline = fgetl(fid);
    while ischar(tline)
        % Ignore the line if it starts with % or if it's empty
        if ~startsWith(tline, '%') && ~isempty(strtrim(tline))
            % Split the line into tokens based on whitespace
            tokens = strsplit(tline);
            
            % Convert numeric tokens to numbers
            for i = 1:length(tokens)
                num = str2double(tokens{i});
                if ~isnan(num)
                    tokens{i} = num;
                end
            end
            
            % Add the tokens to the cell array
            params{end+1} = tokens;
        end
        
        % Read the next line
        tline = fgetl(fid);
    end
    
    % Close the file
    fclose(fid);
end

function [file1, file2, folder, fig] = fileSelector(lines)
    % Create the figure with increased width
%     fig = uifigure('Name', 'File Selector', 'Position', [100 100 450 300]);
% 
%     % Create the labels
%     label1 = uilabel(fig, 'Position', [50 250 200 22], 'Text', 'Select configuration file');
%     label2 = uilabel(fig, 'Position', [50 160 200 22], 'Text', 'Select params file');
%     label3 = uilabel(fig, 'Position', [50 80 200 22], 'Text', 'Select folder to save the data (optional: set if saveBool = 1)');
% 
%     % Create the text boxes
%     txt1 = uitextarea(fig, 'Position', [50 230 300 60], 'Editable', 'off','Value',lines{1});
%     txt2 = uitextarea(fig, 'Position', [50 140 300 60], 'Editable', 'off','Value',lines{2});
%     txt3 = uitextarea(fig, 'Position', [50 60 300 60], 'Editable', 'off','Value',lines{3});
% 
%     % Create the buttons
%     btn1 = uibutton(fig, 'Position', [370 230 70 22], 'Text', 'Select file', ...
%         'ButtonPushedFcn', @(btn1,event) selectFile(txt1, '*.cfg'));
%     btn2 = uibutton(fig, 'Position', [370 180 70 22], 'Text', 'Select file', ...
%         'ButtonPushedFcn', @(btn2,event) selectFile(txt2, '*.txt'));
%     btn3 = uibutton(fig, 'Position', [370 130 70 22], 'Text', 'Select', ...
%         'ButtonPushedFcn', @(btn3,event) selectFolder(txt3));
%     btn4 = uibutton(fig, 'Position', [175 90 100 22], 'Text', 'Done', ...
%         'ButtonPushedFcn', @(btn4,event) done());

    % Create the figure with increased width
    fig = uifigure('Name', 'File Selector', 'Position', [100 100 450 300]);
    
    % Create the labels
    label1 = uilabel(fig, 'Position', [50 280 200 22], 'Text', 'Select configuration file');
    label2 = uilabel(fig, 'Position', [50 190 200 22], 'Text', 'Select params file');
    label3 = uilabel(fig, 'Position', [50 110 300 22], 'Text', 'Select folder to save the data (optional)');
    
    % Create the text boxes (increased height)
    txt1 = uitextarea(fig, 'Position', [50 230 300 50], 'Editable', 'off', 'Value', lines{1});
    txt2 = uitextarea(fig, 'Position', [50 140 300 50], 'Editable', 'off', 'Value', lines{2});
    txt3 = uitextarea(fig, 'Position', [50 60 300 50], 'Editable', 'off', 'Value', lines{3});
    
    % Create the buttons
    btn1 = uibutton(fig, 'Position', [370 230 70 22], 'Text', 'Select file', ...
        'ButtonPushedFcn', @(btn1,event) selectFile(txt1, '*.cfg'));
    btn2 = uibutton(fig, 'Position', [370 140 70 22], 'Text', 'Select file', ...
        'ButtonPushedFcn', @(btn2,event) selectFile(txt2, '*.txt'));
    btn3 = uibutton(fig, 'Position', [370 60 70 22], 'Text', 'Select', ...
        'ButtonPushedFcn', @(btn3,event) selectFolder(txt3));
    btn4 = uibutton(fig, 'Position', [175 30 100 22], 'Text', 'Done', ...
        'ButtonPushedFcn', @(btn4,event) done());


    % Initialize the last directory to the default directory
    lastDir = pwd;

    % Wait for the figure to close
    uiwait(fig);

    % Get the file paths
    file1 = txt1.Value;
    file2 = txt2.Value;
    folder = txt3.Value;

    % Nested function to select a file
    function selectFile(txt, ext)
        [file, path] = uigetfile(fullfile(lastDir, ext));
        if isequal(file, 0)
            disp('User selected Cancel');
        else
            txt.Value = fullfile(path, file);
            lastDir = path;  % Update the last directory
        end
    end

    % Nested function to select a folder
    function selectFolder(txt)
        folder = uigetdir(lastDir);
        if isequal(folder, 0)
            disp('User selected Cancel');
        else
            txt.Value = folder;
            lastDir = folder;  % Update the last directory
        end
    end

    % Nested function to close the figure
    function done()
        uiresume(fig);
    end
end

function adjacencyMatrix = combineClusters(CC, imageSize, binaryImage)
    num = numel(CC);  % Number of clusters
    adjacencyMatrix = false(num);  % Initialize adjacency matrix
    se = strel('square', 3);  % Define structuring element for dilation

    for i = 1:num
        for j = i+1:num
            % Create binary images for the two clusters
            cluster1 = false(imageSize);
            cluster1(CC{i}) = true;
            cluster2 = false(imageSize);
            cluster2(CC{j}) = true;

            % Dilate cluster1 and check if it overlaps with cluster2
            dilatedCluster1 = imdilate(cluster1, se);
            dilatedCluster2 = imdilate(cluster2, se);

%             figure(3)
%             subplot(2,3,1); imshow(cluster1)
%             subplot(2,3,2); imshow(cluster2)
%             subplot(2,3,3); imshow(dilatedCluster1)
%             subplot(2,3,4); imshow(dilatedCluster2)
%             subplot(2,3,5); imshow(dilatedCluster1 & dilatedCluster2)
%             subplot(2,3,6); imshow(binaryImage)

            if any(any(dilatedCluster1 & dilatedCluster2))
                adjacencyMatrix(i, j) = true;
                adjacencyMatrix(j, i) = true;  % The adjacency matrix is symmetric
            end
        end
    end
end

function adjacencyMatrix = combineClustersAggressive(CC, imageSize, binaryImage, maxAggressiveness)
    num = numel(CC);  % Number of clusters
    adjacencyMatrix = false(num);  % Initialize adjacency matrix

    for i = 1:num
        for j = i+1:num
            % Create binary images for the two clusters
            cluster1 = false(imageSize);
            cluster1(CC{i}) = true;
            cluster2 = false(imageSize);
            cluster2(CC{j}) = true;

            % Determine the aggressiveness based on the vertical position of the blobs
            [y1, ~] = ind2sub(imageSize, CC{i});  % Convert indices to coordinates
            [y2, ~] = ind2sub(imageSize, CC{j});  % Convert indices to coordinates
            centroid1 = mean(y1);  % Compute the centroid of cluster1
            centroid2 = mean(y2);  % Compute the centroid of cluster2
            aggressiveness1 = maxAggressiveness * (1 - centroid1 / imageSize(1));
            aggressiveness2 = maxAggressiveness * (1 - centroid2 / imageSize(1));

            % Define structuring elements for dilation
            se1 = strel('disk', round(aggressiveness1), 0);
            se2 = strel('disk', round(aggressiveness2), 0);

            % Dilate clusters and check if they overlap
            dilatedCluster1 = imdilate(cluster1, se1);
            dilatedCluster2 = imdilate(cluster2, se2);

%             figure;
%             subplot(2,3,1); imshow(cluster1)
%             subplot(2,3,2); imshow(cluster2)
%             subplot(2,3,3); imshow(dilatedCluster1)
%             subplot(2,3,4); imshow(dilatedCluster2)
%             subplot(2,3,5); imshow(dilatedCluster1 & dilatedCluster2)
%             subplot(2,3,6); imshow(binaryImage)

            if any(any(dilatedCluster1 & dilatedCluster2))
                adjacencyMatrix(i, j) = true;
                adjacencyMatrix(j, i) = true;  % The adjacency matrix is symmetric
            end
        end
    end
end

function adjacencyMatrix = combineClustersAggressiveRect(CC, imageSize, binaryImage, maxAggressivenessRow, maxAggressivenessCol)
    num = numel(CC);  % Number of clusters
    adjacencyMatrix = false(num);  % Initialize adjacency matrix
    midCol = round(imageSize(2)/2);  % Middle column

    for i = 1:num
        for j = i+1:num
            % Create binary images for the two clusters
            cluster1 = false(imageSize);
            cluster1(CC{i}) = true;
            cluster2 = false(imageSize);
            cluster2(CC{j}) = true;

            % Determine the aggressiveness based on the vertical position of the blobs
            [y1, x1] = ind2sub(imageSize, CC{i});  % Convert indices to coordinates
            [y2, x2] = ind2sub(imageSize, CC{j});  % Convert indices to coordinates
            centroid1 = mean(y1);  % Compute the centroid of cluster1
            centroid2 = mean(y2);  % Compute the centroid of cluster2
            aggressivenessRow1 = maxAggressivenessRow * (1 - centroid1 / imageSize(1));
            aggressivenessRow2 = maxAggressivenessRow * (1 - centroid2 / imageSize(1));
            aggressivenessCol1 = maxAggressivenessCol * (1 - abs(midCol - mean(x1)) / midCol);
            aggressivenessCol2 = maxAggressivenessCol * (1 - abs(midCol - mean(x2)) / midCol);

            % Define structuring elements for dilation
            se1 = strel('rectangle', [round(aggressivenessRow1), round(aggressivenessCol1)]);
            se2 = strel('rectangle', [round(aggressivenessRow2), round(aggressivenessCol2)]);

            % Dilate clusters and check if they overlap
            dilatedCluster1 = imdilate(cluster1, se1);
            dilatedCluster2 = imdilate(cluster2, se2);

            if any(any(dilatedCluster1 & dilatedCluster2))
                adjacencyMatrix(i, j) = true;
                adjacencyMatrix(j, i) = true;  % The adjacency matrix is symmetric
            end
        end
    end
end

function img = plotClusters(validClusters, otherValidClusters, adjacencyMatrix, imageSize)
    % Initialize the output image
    img = false(imageSize);
    
    % Loop over each cluster
    for i = 1:numel(validClusters)
        % Create a binary image for the cluster
        cluster = false(imageSize);
        cluster(validClusters{i}) = true;
        
        % If the cluster is adjacent to any other cluster, dilate it
        if any(adjacencyMatrix(i, :))
            se = strel('square', 3);  % Define structuring element for dilation
            cluster = imdilate(cluster, se);
        end
        
        % Add the cluster to the output image
        img = img | cluster;
    end

    for i = 1:numel(otherValidClusters)
        % Create a binary image for the cluster
        cluster = false(imageSize);
        cluster(otherValidClusters{i}) = true;
        
        % Add the cluster to the output image
        img = img | cluster;
    end
    img = ~img;
end

function img = watershedByParts(heatmapPart, heatmap, hmin, otsuMultiplier)
%     subplot(1,3,1)
%     surf(heatmapPart); view(2);
%     subplot(1,3,2)
%     surf(heatmap); view(2);
    minVal = 0;
    maxVal = max(heatmapPart(:));
    
    % Normalize the matrix to [0, 1]
    normalizedMatrix = (heatmapPart - minVal) / (maxVal - minVal);
    ms = 1 - normalizedMatrix(:,:);
    ms = uint8(ms * 255);

    hs = watershed(imhmin(ms,hmin));          
    num_segments = cast(max(hs(:)),'int8');

    final_segmanted = zeros(size(ms));
    % Loop over each segment
    for k = 1:num_segments
        % Create a binary mask for the current segment
        mask = hs == k;
        
        % Apply the mask to the image
        segment = ms .* uint8(mask);
        
        % Apply Otsu's thresholding to the segment
        level = graythresh(segment(mask));
        bw = imbinarize(segment, level*otsuMultiplier);

        % Store the thresholded segment in the cell array
        final_segmanted = final_segmanted | bw;

        % Find the connected components in the thresholded segment
        CC = bwconncomp(~(bw | ~mask));
    end
    img = hs==0 | final_segmanted;
%     subplot(1,3,3)
%     imshow(flip(hs==0 | final_segmanted))
end





