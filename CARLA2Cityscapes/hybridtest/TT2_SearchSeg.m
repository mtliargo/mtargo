addpath util/; Platform;

runStr = 'test1';
nTestMax = 1;
bVisual = 1;

segBankDir = '../../SIMS/traindata/original_segment';
segListDir = fullfile(dataDir, 'Exp/C2C', runStr, 'SegList-256');
matchDir = mkdir2(fullfile(dataDir, 'Exp/C2C', runStr, 'SegMatch-256'));
if bVisual
    visDir = mkdir2([matchDir '-Vis']);
    imgBankDir = fullfile(dataDir, 'Cityscapes/ReOrg/Image-256/train');
end

classList = dir(fullfile(segListDir));
classList = {classList.name};
classList = classList(~ismember(classList, {'.', '..'}));

nClass = length(classList);

for c = 1:nClass
    classDir = fullfile(segListDir, classList{c});
    segBankClassDir = fullfile(segBankDir, classList{c});
    outDir = mkdir2(fullfile(matchDir, classList{c}));
    testList = dir(fullfile(classDir, '*.mat'));
    testList = {testList.name};
    nTest = min(nTestMax, length(testList));
    for i = 1:nTest
        ldata = load(fullfile(classDir, testList{i}));
        [match, iou, originIdx] = matchseg(ldata.library_mask, segBankClassDir);
        out = struct;
        out.proposal = match;
        out.proposal_iou_all = iou;
        out.response_index = originIdx;
        sz = size(match);
        if length(sz) < 5
            sz(5) = 1;
        end
        % 1 means top 1 here
        out.proposal_pole_mask = zeros([sz(1) sz(2) 1 sz(5)], 'uint8');
        saveseg(fullfile(outDir, testList{i}), out);
    end
end 

if bVisual
    classDir = fullfile(segListDir, classList{1});
    testList = dir(fullfile(classDir, '*.mat'));
    testList = {testList.name};
    nTest = min(nTestMax, length(testList));
    visseg(testList(1:nTest), classList, visDir, segListDir, matchDir, imgBankDir);
end





