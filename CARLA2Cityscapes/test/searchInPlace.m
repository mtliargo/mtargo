function [match, source, iou] = searchInPlace(mask, classIdx, segBankDir, topK)

bGPU = 0;
minIoU = 0;

n = length(mask);
match = cell(n, topK);
source = zeros(n, topK);
iou = zeros(n, topK);

assert(topK == 1); % topK = 1 for now

mask = cat(3, mask{:});
if bGPU
    mask = gpuArray(mask);
    iou = gpuArray(iou);
end

classList = dir(fullfile(segBankDir, '*.mat'));
classList = {classList.name};

for c = 1:length(classList)
    thisClass = find(classIdx == str2double(classList{c}(1:end-4)));
    nThisClass = length(thisClass);
    thisClassMask = mask(:, :, thisClass);
    
    ldata = load(fullfile(segBankDir, classList{c}));
    bankMask = ldata.mask;
    bankSource = ldata.source;
    bankSize = length(bankMask);
    bankMask = cat(3, bankMask{:});
    
    if bGPU
        bankMask = gpuArray(bankMask);
    end
   
    for i = 1:nThisClass
        maxIoU = 0;
        for j = 1:bankSize
            itsc = thisClassMask(:, :, i) & bankMask(:, :, j);
            u = thisClassMask(:, :, i) | bankMask(:, :, j);
            thisIoU = sum(itsc(:)) / sum(u(:));
            if thisIoU > max(minIoU, maxIoU)
                iou(thisClass(i), 1) = thisIoU;
                match{thisClass(i), 1} = bankMask(:, :, j);
                source(thisClass(i), 1) = bankSource(j);
            end
        end
    end
end

if bGPU
    iou = gather(iou);
    match = cellfun(@gather, match);
end

end





