function data = preprocessData(data, targetSize)
    for ii = 1:size(data,1)
        I = data{ii,1};
        imgSize = size(I);
        bboxes = data{ii,2};
        I = im2single(imresize(I, targetSize(1:2)));
        scale = targetSize(1:2) ./ imgSize(1:2);
        bboxes = bboxresize(bboxes, scale);
        data(ii,1:2) = {I, bboxes};
    end
end
