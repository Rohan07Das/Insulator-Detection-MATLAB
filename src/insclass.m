%% STEP 1: Load Cropped Dataset
imds = imageDatastore('C:/yolo/croppedInsulators', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Filter small images
validIdx = false(numel(imds.Files), 1);
for i = 1:numel(imds.Files)
    img = imread(imds.Files{i});
    if size(img,1) > 40 && size(img,2) > 40
        validIdx(i) = true;
    end
end
imds = subset(imds, find(validIdx));

fprintf("✅ %d images across %d classes\n", numel(imds.Files), numel(categories(imds.Labels)));
disp(countEachLabel(imds));

% Split into training and validation sets
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

%% STEP 2: Load Pretrained SqueezeNet and Modify
net = squeezenet;
lgraph = layerGraph(net);

% Remove old classification layers
lgraph = removeLayers(lgraph, {'conv10','relu_conv10','pool10','prob','ClassificationLayer_predictions'});

% Add new classification layers
numClasses = numel(categories(imds.Labels));
newLayers = [
    convolution2dLayer(1, numClasses, 'Name','new_conv', ...
        'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10)
    reluLayer('Name','new_relu')
    globalAveragePooling2dLayer('Name','new_gap')
    softmaxLayer('Name','new_softmax')
    classificationLayer('Name','new_output')
];

% Add and connect
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'drop9', 'new_conv');

%% STEP 3: Prepare Data (Resize + RGB Conversion)
inputSize = net.Layers(1).InputSize;

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'ColorPreprocessing','gray2rgb');

augimdsVal = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing','gray2rgb');

%% STEP 4: Define Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsVal, ...
    'ValidationPatience',5, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% STEP 5: Load Trained Classifier and Setup
load('C:/yolo/insulatorSqueezeNetClassifier.mat', 'trainedClassifier');  % 👈 your classifier
inputSize = trainedClassifier.Layers(1).InputSize;

% Load trained YOLO detector
load('C:/yolo/insulatorYOLOv4.mat', 'detector');

% Choose a test image
testImage = imread("C:\yolo\insulators\images\031.jpg");  % 👈 change path

% Run detection
[bboxes, scores, labels] = detect(detector, testImage);

% Classification
sqzLabels = strings(size(bboxes,1),1);  % classified labels
results = table;                        % table to collect results
confidenceThreshold = 0.5;

for i = 1:size(bboxes,1)
    % Skip low-confidence detections
    if scores(i) < confidenceThreshold
        continue;
    end

    % Crop and resize
    crop = imcrop(testImage, bboxes(i,:));
    crop = imresize(crop, inputSize(1:2));

    % Ensure 3 channels
    if size(crop,3) == 1
        crop = repmat(crop, [1 1 3]);
    end

    % Classify
    sqzLabel = classify(trainedClassifier, crop);
    sqzLabels(i) = string(sqzLabel);

    % Store result
    results = [results;
        table(bboxes(i,:), string(labels(i)), sqzLabels(i), ...
        'VariableNames', {'BBox', 'YOLOLabel', 'SqueezeNetLabel'})];

    % Optional: Display each crop
    figure; imshow(crop);
    yoloLabel = strrep(string(labels(i)), "_", "\_");
    sqzLabel  = strrep(sqzLabels(i), "_", "\_");
    title(sprintf("YOLO: %s | SqueezeNet: %s", yoloLabel, sqzLabel));
end

disp("✅ YOLO + SqueezeNet classification results:");
disp(results);


%% STEP 6: Confusion Matrix and Match Accuracy

% Filter rows where classification was performed
validRows = sqzLabels ~= "";

% Extract YOLO and SqueezeNet labels
trueLabels = string(results.YOLOLabel(validRows));        % Ground truth from YOLO
predLabels = string(results.SqueezeNetLabel(validRows));  % Predicted class by SqueezeNet

% Plot confusion matrix
figure;
confusionchart(trueLabels, predLabels, ...
    'Title', 'YOLO vs SqueezeNet Confusion Matrix (Insulator)', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

% Compute match accuracy (YOLO label == SqueezeNet label)
matchAccuracy = sum(trueLabels == predLabels) / numel(trueLabels);
fprintf("✅ SqueezeNet vs YOLO Match Accuracy: %.2f%%\n", matchAccuracy * 100);
