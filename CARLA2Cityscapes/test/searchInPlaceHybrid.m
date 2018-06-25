function [match, sourceIdx, iou] = searchInPlaceHybrid(masks, classes, SIMSBankDir, topK)

bGPU = true;
minIoU = 0.2;

n = length(masks);
match = cell(n, topK);
sourceIdx = zeros(n, topK);
iou = zeros(n, topK);

assert(topK == 1); % topK = 1 for now

classList = dir(SIMSBankDir);
classList = {classList.name};
classList = classList(~ismember(classList, {'.', '..'}));

masks = cat(3, masks{:});
if bGPU
    masks = gpuArray(masks);
    iou = gpuArray(iou);
end

for c = 1:length(classList)
    thisClass = find(classes == str2double(classList{c}));
    nThisClass = length(thisClass);
    thisClassMasks = masks(:, :, thisClass);
    classDir = fullfile(SIMSBankDir, classList{c});
    bankList = dir(fullfile(classDir, '*.mat'));
    bankList = {bankList.name};
    
    intsec = zeros(nThisClass, 1, 'gpuArray');
    for i = 1:length(bankList)
        ldata = load(fullfile(classDir, bankList{i}));
        segs = gpuArray(ldata.library_mask);
        nSeg = size(segs, 3);
        for m = 1:nThisClass
            for s = 1:nSeg
                itsc = thisClassMasks(:, :, m) & segs(:, :, s);
                itsc = sum(itsc(:));
                if itsc > intsec(m)
                    intsec(m) = itsc;
                    u = thisClassMasks(:, :, m) | segs(:, :, s);
                    thisIoU = itsc / sum(u(:));
                    if thisIoU > minIoU
                        iou(thisClass(m), 1) = gather(thisIoU);
                        match{thisClass(m), 1} = gather(segs(:, :, s));
                        sourceIdx(thisClass(m), 1) = i;
                    end
                end
            end
        end
    end
end

if bGPU
    iou = gather(iou);
    match = cellfun(@gather, match, 'UniformOutput', false);
end

end



