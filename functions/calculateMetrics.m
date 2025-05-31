function calculateMetrics(trainedNet, imdsTest)
    % Giriş boyutu
    inputSize = trainedNet.Layers(1).InputSize;
    
    % 👇 Burada gri resimleri RGB'ye çevirdik, yoksa patlıyor
    imdsTestResized = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
        'ColorPreprocessing', 'gray2rgb');

    % Tahmin et
    predictedLabels = classify(trainedNet, imdsTestResized);
    trueLabels = imdsTest.Labels;

    % Confusion Matrix
    confMat = confusionmat(trueLabels, predictedLabels);

    if size(confMat,1) == 2 && size(confMat,2) == 2
        TP = confMat(1,1);
        FN = confMat(1,2);
        FP = confMat(2,1);
        TN = confMat(2,2);
    else
        error('İki sınıflı problem için geçerli confusion matrix elde edilemedi.');
    end

    % Metrikler
    Accuracy = (TP + TN) / (TP + TN + FP + FN);
    Precision = TP / (TP + FP);
    Sensitivity = TP / (TP + FN); % Recall
    F1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity);

    % Yazdır
    fprintf('📊 Model Performansı:\n');
    fprintf('Accuracy   : %.2f %%\n', Accuracy);
    fprintf('Precision  : %.2f %%\n', Precision);
    fprintf('Sensitivity: %.2f %%\n', Sensitivity );
    fprintf('F1-Score   : %.2f %%\n', F1 );
end
