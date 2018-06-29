addpath ../../util/; Platform;

runStr = 'test5';
nTestMax = 30;
bVisual = 1;
topK = 1;

segBankDir = fullfile(dataDir, 'Exp/C2C/bank1/');
segListDir = fullfile(dataDir, 'Exp/C2C', runStr, 'SegList-256');
matchDir = fullfile(dataDir, 'Exp/C2C', runStr, 'SegMatch-256');

visDir = mkdir2([matchDir '-Vis']);
visExDir = mkdir2([matchDir '-VisEx']);
visExSuffix = {
    '_road1.jpg';
    '_car1.jpg';
    '_car2.jpg';
    '_car3.jpg';
    '_building1.jpg';
};
imgBankDir = fullfile(dataDir, 'Cityscapes/ReOrg/Image-256/train');


testList = dir(fullfile(segListDir, '*.mat'));
testList = {testList.name};
nTest = min(nTestMax, length(testList));
for i = 1:nTest
    ldata = load(fullfile(segListDir, testList{i}));
    mask = ldata.mask;
    classIdx = ldata.classIdx;
    ldata = load(fullfile(matchDir, testList{i}));
    match = ldata.match;
    source = ldata.source;
    
    [img, orgImgs] = visSearch(mask, classIdx, match, source, imgBankDir);
    imwrite(img, fullfile(visDir, [testList{i}(1:end-3) 'jpg']));
    for j = 1:length(visExSuffix)
        if isempty(orgImgs{j})
            imgEx = zeros(size(img), 'like', img);
        else
            imgEx = orgImgs{j};
        end
        imwrite(imgEx, fullfile(visExDir, [testList{i}(1:end-4) visExSuffix{j}]));
    end
end







