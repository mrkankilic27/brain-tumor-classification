
function brain_tumor_gui_only
    clc; close all;

    % 1. Custom CNN Modelini Yükle
    try
        temp1 = load('trained_custom_model.mat');
        fields1 = fieldnames(temp1);
        customNet = temp1.(fields1{1});
    catch
        [file1, path1] = uigetfile('*.mat', 'Custom Modeli Seç');
        if isequal(file1, 0)
            error('Custom model seçilmedi.');
        end
        temp1 = load(fullfile(path1, file1));
        fields1 = fieldnames(temp1);
        customNet = temp1.(fields1{1});
    end

    % 2. GoogLeNet Modelini Yükle
    try
        temp2 = load('trained_googlenet_model.mat');
        fields2 = fieldnames(temp2);
        googlenetTrained = temp2.(fields2{1});
    catch
        [file2, path2] = uigetfile('*.mat', 'GoogLeNet Modelini Seç');
        if isequal(file2, 0)
            error('GoogLeNet modeli seçilmedi.');
        end
        temp2 = load(fullfile(path2, file2));
        fields2 = fieldnames(temp2);
        googlenetTrained = temp2.(fields2{1});
    end

    % 3. GUI Arayüzü
    fig = uifigure('Name', 'Brain Tumor Classifier', 'Position', [100 100 800 600]);
    ax = uiaxes(fig, 'Position', [50 150 700 400], 'Box', 'on');
    title(ax, 'MRI Image Display');

    % Model Seçimi
    modelSelector = uidropdown(fig, ...
        'Position', [50 100 200 30], ...
        'Items', {'Custom CNN', 'GoogLeNet'}, ...
        'Value', 'Custom CNN', ...
        'FontSize', 12);

    % Eğitim Optimizasyonu Seçimi
    optimizerSelector = uidropdown(fig, ...
        'Position', [300 100 200 30], ...
        'Items', {'adam', 'sgdm', 'rmsprop'}, ...
        'Value', 'sgdm', ...
        'FontSize', 12);

    % Kalite Seçimi
    qualitySelector = uidropdown(fig, ...
        'Position', [550 100 200 30], ...
        'Items', {'High Quality', 'Low Quality'}, ...
        'Value', 'High Quality', ...
        'FontSize', 12);

    % Sınıflandırma Butonu
    classifyBtn = uibutton(fig, 'push', ...
        'Position', [550 50 150 30], ...
        'Text', 'Classify Image', ...
        'FontSize', 12, ...
        'ButtonPushedFcn', @(btn, event) classifyImage(ax, modelSelector.Value, optimizerSelector.Value, qualitySelector.Value, customNet, googlenetTrained));

    % Eğitim Butonu (Optimizasyon ve Kalite ile)
    trainBtn = uibutton(fig, 'push', ...
        'Position', [50 50 200 30], ...
        'Text', 'Train Models', ...
        'FontSize', 12, ...
        'ButtonPushedFcn', @(btn, event) onTrainBtnPushed());

    % Eğitim buton fonksiyonu
    function onTrainBtnPushed()
        [updatedCustomNet, updatedGoogLeNet] = trainModels(optimizerSelector.Value, qualitySelector.Value, customNet, googlenetTrained);
        customNet = updatedCustomNet;
        googlenetTrained = updatedGoogLeNet;
        uialert(fig, 'Modeller başarıyla yeniden eğitildi.', 'Başarılı');
    end
end

% Sınıflandırma Fonksiyonu
function classifyImage(ax, modelType, optimizer, quality, customNet, googlenetTrained)
    [file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'});
    if isequal(file, 0)
        return;
    end

    img = imread(fullfile(path, file));
    if size(img,3) == 1
        img = repmat(img, [1 1 3]);
    end
    img = imresize(img, [224 224]);
    imshow(img, 'Parent', ax);

    try
        if strcmp(modelType, 'Custom CNN')
            [label, scores] = classify(customNet, img);
        else
            [label, scores] = classify(googlenetTrained, img);
        end
        msgbox(['Tahmin: ', char(label), ' (Güven: ', num2str(max(scores)*100, '%.2f'), '%)']);
    catch e
        errordlg(['Hata: ' e.message]);
    end
end

% Eğitim Fonksiyonu (Optimizasyon ve Kalite Seçeneklerine Göre)
function [updatedCustomNet, updatedGoogLeNet] = trainModels(optimizer, quality, customNet, googlenetTrained)
    datasetPath = fullfile('brain_tumor_dataset');
    imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    [imdsTrain, imdsValidation, ~] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

    augmenter = imageDataAugmenter('RandRotation', [-20 20], 'RandXReflection', true, 'RandYReflection', true);
    augmentedImdsTrain = augmentedImageDatastore([224 224], imdsTrain, 'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
    augmentedImdsValidation = augmentedImageDatastore([224 224], imdsValidation, 'ColorPreprocessing', 'gray2rgb');

    if strcmp(quality, 'High Quality')
        miniBatchSize = 32;
        maxEpochs = 15;
    else
        miniBatchSize = 16;
        maxEpochs = 10;
    end

    options = trainingOptions(optimizer, ...
        'InitialLearnRate', 0.001, ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'ValidationData', augmentedImdsValidation, ...
        'ValidationFrequency', 30, ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto');

    disp(['Training Custom CNN with optimizer: ', optimizer, ' and ', quality, ' quality.']);
    customNet = trainNetwork(augmentedImdsTrain, customNet.Layers, options);

    disp(['Training GoogLeNet with optimizer: ', optimizer, ' and ', quality, ' quality.']);
    googlenetTrained = trainNetwork(augmentedImdsTrain, googlenetTrained.Layers, options);

    save('trained_custom_model.mat', 'customNet');
    save('trained_googlenet_model.mat', 'googlenetTrained');

    updatedCustomNet = customNet;
    updatedGoogLeNet = googlenetTrained;

    disp('Models training complete and saved.');
end
