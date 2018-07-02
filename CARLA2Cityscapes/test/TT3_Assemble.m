addpath ../../util/; Platform;
addpath ../dbcode/;

runStr = 'test6';
nTestMax = 30;
inSize = 256;
outSize = 512;
bVisual = 1;

classColor = uint8(dlmread('../dbcode/classColor.txt'));

segListDir = fullfile(dataDir, 'Exp/C2C', runStr, ['SegList-' num2str(inSize)]);
matchDir = fullfile(dataDir, 'Exp/C2C', runStr, ['SegMatch-' num2str(inSize)]);

imgBankDir = fullfile(dataDir, 'Cityscapes/ReOrg', ['Image-' num2str(outSize)], 'train');
segColorDir = fullfile(dataDir, 'Exp/C2C', runStr, ['SegColor-' num2str(outSize)]);

outDir = mkdir2(fullfile(dataDir, 'Exp/C2C', runStr, ['Canvas-' num2str(outSize)]));
visDir = mkdir2([outDir '-Vis']);

testList = dir(fullfile(segListDir, '*.mat'));
testList = {testList.name};
nTest = min(nTestMax, length(testList));
for i = 1:nTest
    ldata = load(fullfile(segListDir, testList{i}));
    mask = ldata.mask;
    classIdx = ldata.classIdx;
    ldata = load(fullfile(matchDir, testList{i}));
    source = ldata.source;
    
    segColor = imread(fullfile(segColorDir, [testList{i}(1:end-3) 'png']));
    idMap = uint8(color2id(segColor, classColor));
    
    canvas = assembleSeg(mask, classIdx, source, imgBankDir, idMap, outSize);
    % Convert to SIMS format
    %    proposal: [512×1024×3 single] from 0 to 1
    %    label: [512×1024×3 uint8]
    
    proposal = im2single(canvas);
    label = segColor;
    save(fullfile(outDir, testList{i}), 'proposal', 'label');
    if bVisual
        imwrite(canvas, fullfile(visDir, [testList{i}(1:end-3) 'jpg']));
    end
end









