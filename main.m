%% Brain Tumor Classification - Enhanced Version with GoogLeNet and Custom CNN
clc; clear; close all;

%% 1. Data Loading and Preprocessing
disp('Loading data...');
datasetPath = fullfile('brain_tumor_dataset');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% 2. Data Splitting
disp('Splitting data...');
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');


%% 3. Data Augmentation
disp('Preparing data augmentation...');
augmenter = imageDataAugmenter(... 
    'RandRotation', [-20 20],...
    'RandXReflection', true,...
    'RandYReflection', true);

augmentedImdsTrain = augmentedImageDatastore([224 224], imdsTrain,...
    'DataAugmentation', augmenter,...
    'ColorPreprocessing', 'gray2rgb');

augmentedImdsValidation = augmentedImageDatastore([224 224], imdsValidation,...
    'ColorPreprocessing', 'gray2rgb');

augmentedImdsTest = augmentedImageDatastore([224 224], imdsTest,...
    'ColorPreprocessing', 'gray2rgb');

%% 4. Model Creation
disp('Building models...');
% 4.1 Custom CNN Model
customLayers = [... 
    imageInputLayer([224 224 3], 'Name', 'input')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(2, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

% 4.2 GoogLeNet Pretrained Model
if ~exist('googlenet.mat', 'file')
    disp('Downloading GoogLeNet...');
    net = googlenet;
    save('googlenet.mat', 'net');
else
    loadedData = load('googlenet.mat');
    if isfield(loadedData, 'net')
        net = loadedData.net;
    elseif isfield(loadedData, 'googlenet')
        net = loadedData.googlenet;
    else
        error('Loaded MAT file does not contain a valid network named "net" or "googlenet".');
    end
end

lgraph = layerGraph(net);
newLayers = [... 
    fullyConnectedLayer(2, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classoutput')];

lgraph = replaceLayer(lgraph, 'loss3-classifier', newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));
lgraph = replaceLayer(lgraph, 'output', newLayers(3));

%% 5. Training Options
optimizers = {'rmsprop', 'adam', 'sgdm'};
qualities = {'High Quality', 'Low Quality'};
models = {'Custom CNN', 'GoogLeNet'};

%% 6. Model Training for All Variations
disp('Training models with different configurations...');
for i = 1:length(models)
    for j = 1:length(optimizers)
        for k = 1:length(qualities)
            disp(['Training ', models{i}, ' with optimizer ', optimizers{j}, ' and quality ', qualities{k}]);

            if strcmp(qualities{k}, 'High Quality')
                miniBatchSize = 32;
                maxEpochs = 30;
            else
                miniBatchSize = 16;
                maxEpochs = 15;
            end

            options = trainingOptions(optimizers{j},...
                'InitialLearnRate', 0.0001,...
                'MaxEpochs', maxEpochs,...
                'MiniBatchSize', miniBatchSize,...
                'ValidationData', augmentedImdsValidation,...
                'ValidationFrequency', 30,...
                'Verbose', true,...
                'Plots', 'training-progress',...
                'ExecutionEnvironment', 'auto');

            if strcmp(models{i}, 'Custom CNN')
                trainedNet = trainNetwork(augmentedImdsTrain, customLayers, options);
                modelName = ['trained_custom_' optimizers{j} '_' qualities{k} '.mat'];
                save(modelName, 'trainedNet');
                disp(['Saved model: ', modelName]);
            elseif strcmp(models{i}, 'GoogLeNet')
                trainedNet = trainNetwork(augmentedImdsTrain, lgraph, options);
                modelName = ['trained_googlenet_' optimizers{j} '_' qualities{k} '.mat'];
                save(modelName, 'trainedNet');
                disp(['Saved model: ', modelName]);
            end

            % Tahmin ve Konfüzyon Matrisi
            YPred = classify(trainedNet, augmentedImdsTest);
            accuracy = sum(YPred == imdsTest.Labels) / numel(imdsTest.Labels);
            disp([models{i}, ' Test Accuracy: ', num2str(accuracy * 100), '%']);
            calculateMetrics(trainedNet, imdsTest);
            figure;
            confusionchart(imdsTest.Labels, YPred, ...
                'Title', [models{i}, ' - ', optimizers{j}, ' - ', qualities{k}, ' - Confusion Matrix']);
        end
    end
end

%% 7. Save Final Models for GUI
disp('Saving final models...');
if exist('trainedNetworks', 'dir') == 0
    mkdir('trainedNetworks');
end

% Son kullanılan eğitimli modelleri kaydet
if exist('trainedNet', 'var')
    if isa(trainedNet.Layers(1), 'nnet.cnn.layer.ImageInputLayer') % Custom CNN kontrolü
        trainedCNN = trainedNet;
        save(fullfile('trainedNetworks', 'trainedCNN.mat'), 'trainedCNN');
    else % GoogLeNet kontrolü
        googlenetTrained = trainedNet;
        save(fullfile('trainedNetworks', 'trainedGoogLeNet.mat'), 'googlenetTrained');
    end
end

disp('All selected models saved successfully.');