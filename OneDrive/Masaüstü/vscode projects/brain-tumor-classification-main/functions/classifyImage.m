function imageClassifier()
    global gui_net gui_ax gui_lblResult;

    % Select image
    [file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'});
    if isequal(file, 0)
        return;
    end

    % Read image
    img = imread(fullfile(path, file));
    
    % If grayscale image, convert it to RGB
    if size(img,3) == 1
        img = repmat(img, [1 1 3]);
    end
    
    % Resize image to 224x224
    img = imresize(img, [224 224]);

    % Display image in the GUI axes
    imshow(img, 'Parent', gui_ax);

    % Classify image using the selected model
    [label, scores] = classify(gui_net, img);
    
    % Display result
    gui_lblResult.Text = ['Prediction: ', char(label), ...
                         ' (Confidence: ', num2str(max(scores)*100, '%.2f'), '%)'];
end
