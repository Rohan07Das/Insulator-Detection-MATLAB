% === Load dataset and class names ===
load('C:/yolo/insulatorDataset_StringLabels.mat');  % loads 'insulatorDataset'
classNames = strtrim(readlines("C:/yolo/insulators/classes.txt"));

% === Step 1: Convert string columns to cell arrays (if needed) ===
for i = 1:numel(classNames)
    col = classNames(i);
    if isstring(insulatorDataset.(col))
        insulatorDataset.(col) = cellstr(insulatorDataset.(col));  % convert to char cells
    end
end

% === Step 2: Convert each bbox string to numeric Mx4 array ===
for i = 1:numel(classNames)
    col = classNames(i);
    for j = 1:height(insulatorDataset)
        str = insulatorDataset{j, col}{1};
        if isempty(str) || strcmp(str, "")
            insulatorDataset{j, col} = {zeros(0, 4)};
        else
            tokens = regexp(str, '\s*;\s*|\s+', 'split');
            nums = str2double(tokens);
            boxes = reshape(nums, 4, [])';  % convert to Mx4 format
            insulatorDataset{j, col} = {boxes};
        end
    end
end


% === Step 3: Split dataset ===
rng(0);  % for reproducibility
shuffledIdx = randperm(height(insulatorDataset));
n = height(insulatorDataset);

idxTrain = 1:round(0.6*n);
idxVal   = round(0.6*n)+1 : round(0.8*n);
idxTest  = round(0.8*n)+1 : n;

trainingDataTbl   = insulatorDataset(shuffledIdx(idxTrain), :);
validationDataTbl = insulatorDataset(shuffledIdx(idxVal), :);
testDataTbl       = insulatorDataset(shuffledIdx(idxTest), :);

% === Step 4: Create datastores ===
imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
bldsTrain = boxLabelDatastore(trainingDataTbl(:, classNames));
trainingData = combine(imdsTrain, bldsTrain);

imdsVal = imageDatastore(validationDataTbl.imageFilename);
bldsVal = boxLabelDatastore(validationDataTbl(:, classNames));
validationData = combine(imdsVal, bldsVal);

imdsTest = imageDatastore(testDataTbl.imageFilename);
bldsTest = boxLabelDatastore(testDataTbl(:, classNames));
testData = combine(imdsTest, bldsTest);

% === Step 5: Save datasets for training ===
save('C:/yolo/insulator_trainingData.mat', 'trainingData');
save('C:/yolo/insulator_validationData.mat', 'validationData');
save('C:/yolo/insulator_testData.mat', 'testData');

disp("✅ Dataset converted and split into train/val/test successfully.");

% === Step 6: Preprocess and Estimate Anchors ===

% Define input size for YOLOv4
inputSize = [320 320 3];

% Preprocess training data (resizing, padding)
augmentedTrainingData = transform(trainingData, @(data) preprocessData(data, inputSize));

% Estimate anchor boxes (using 9 clusters)
[anchors, meanIoU] = estimateAnchorBoxes(augmentedTrainingData, 9);
disp("📦 Estimated anchor boxes (unsorted):");
disp(anchors);
disp("📈 Mean IoU with estimated anchors: " + meanIoU);

% Sort anchors by area (descending)
areas = anchors(:,1) .* anchors(:,2);
[~, idx] = sort(areas, 'descend');
anchors = anchors(idx, :);

% Group into 3 levels (YOLOv4 uses 3 output scales)
anchorBoxes = {anchors(1:3,:); anchors(4:6,:); anchors(7:9,:)};

disp("✅ Anchor boxes grouped for YOLOv4:");
disp(anchorBoxes);


% === Step 7: Define YOLOv4 Detector ===
detector = yolov4ObjectDetector("csp-darknet53-coco", classNames, anchorBoxes, InputSize=inputSize);

% === Step 8: Training Options ===
options = trainingOptions("adam", ...
    InitialLearnRate = 0.001, ...
    MaxEpochs = 10, ...
    MiniBatchSize = 2, ...
    ValidationData = validationData, ...
    Shuffle = "every-epoch", ...
    VerboseFrequency = 20, ...
    CheckpointPath = "C:/yolo/checkpoints", ...
    Plots = "training-progress", ...
    BatchNormalizationStatistics = "moving", ...
    ResetInputNormalization = false);  % ⚠️ important for pretrained backbones

% === Step 10: Load trained detector ===
load('C:/yolo/insulatorYOLOv4.mat', 'detector');

% === Choose any test image from your insulator dataset ===
testImage = imread("C:\yolo\insulators\images\031.jpg");  % 👈 change filename

% === Run detection ===
[bboxes, scores, labels] = detect(detector, testImage);

% === Annotate results on the image ===
Iout = insertObjectAnnotation(testImage, 'rectangle', bboxes, ...
    cellstr(labels) + " " + string(round(scores * 100)) + "%");

% === Display ===
figure;
imshow(Iout);
title("Insulator Detection with Confidence Scores");


% === Load trained detector and test dataset ===
load('C:/yolo/insulatorYOLOv4.mat', 'detector');
load('C:/yolo/insulator_testData.mat', 'testData');

% === Run detection on test dataset ===
detectionResults = detect(detector, testData);

% === Evaluate precision and recall ===
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults, testData);

% === Load and format class names ===
classNames = strtrim(readlines("C:/yolo/insulators/classes.txt"));
classNames = strrep(classNames, "_", "\_");  % For display (escape underscores)

% === Plot Precision-Recall curves ===
figure;
hold on;
legendEntries = cell(numel(ap), 1);

for i = 1:numel(ap)
    plot(recall{i}, precision{i}, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
    legendEntries{i} = sprintf("%s (AP = %.2f)", classNames(i), ap(i));
end

xlabel("Recall");
ylabel("Precision");
title("Precision-Recall Curves for Insulator Detection");
legend(legendEntries, 'Location', 'eastoutside');
xlim([0 1]);
ylim([0 1]);
grid on;
hold off;

% === Display Mean AP
meanAP = mean(ap);
disp("📌 Mean AP across all classes: " + meanAP);

% === Create metrics table
recallVals = cellfun(@(r) r(end), recall);
precisionVals = cellfun(@(p) p(end), precision);

metricsTable = table(classNames, ap(:), recallVals(:), precisionVals(:), ...
    'VariableNames', {'Class', 'AveragePrecision', 'Recall', 'Precision'});

disp(metricsTable);


