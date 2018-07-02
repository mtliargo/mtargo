addpath ../../util/; Platform;

runStr = 'test6';
nTestMax = 30;
bVisual = 1;
topK = 1;

segBankDir = fullfile(dataDir, 'Exp/C2C/bank1/');
segListDir = fullfile(dataDir, 'Exp/C2C', runStr, 'SegList-256');
matchDir = mkdir2(fullfile(dataDir, 'Exp/C2C', runStr, 'SegMatch-256'));
if bVisual
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
    
%     savest = struct;
%     savest.match = match;
%     savest.source = source;
%     savest.score = score;
%     parsave(fullfile(matchDir, testList{i}), savest);
    
    if bVisual
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
end







