addpath ../../util/; Platform;

runStr = 'test4';
nTestMax = 10;
bVisual = 1;
topK = 1;

segBankDir = fullfile(dataDir, 'Exp/C2C/bank1/');
segListDir = fullfile(dataDir, 'Exp/C2C', runStr, 'SegList-256');
matchDir = mkdir2(fullfile(dataDir, 'Exp/C2C', runStr, 'SegMatch-256'));
if bVisual
    visDir = mkdir2([matchDir '-Vis']);
    imgBankDir = fullfile(dataDir, 'Cityscapes/ReOrg/Image-256/train');
end

testList = dir(fullfile(segListDir, '*.mat'));
testList = {testList.name};
nTest = min(nTestMax, length(testList));
for i = 1:nTest
    ldata = load(fullfile(segListDir, testList{i}));
    mask = ldata.mask;
    classIdx = ldata.classIdx;
    [match, source, score] = searchInPlace(mask, classIdx, segBankDir, topK);
    save(fullfile(matchDir, testList{i}), 'match', 'source', 'score');
    if bVisual
        img = visSearch(mask, source, imgBankDir);
        imwrite(img, fullfile(visDir, [testList{i}(1:end-3) 'jpg']));
    end
end







