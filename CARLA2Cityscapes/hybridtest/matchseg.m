function [match, iou, originIdx] = matchseg(masks, segBankDir)

segList = dir(fullfile(segBankDir, '*.mat'));
segList = {segList.name};
sz = size(masks);

if length(sz) < 3
    sz(3) = 1;
end

nMask = sz(3);
if isempty(masks) || nMask == 0
    match = zeros([sz(1:2) 3 1 0]);
    iou = zeros(0, 1);
    originIdx = zeros(0, 1);
    return
end
    

masks = gpuArray(masks);

originIdx = zeros(nMask, 1);
intsec = zeros(nMask, 1, 'gpuArray');
iou = zeros(nMask, 1, 'gpuArray');
match = zeros(size(masks), 'gpuArray');

for i = 1:length(segList)
    ldata = load(fullfile(segBankDir, segList{i}));
    segs = gpuArray(ldata.library_mask);
    nSeg = size(segs, 3);
    for m = 1:nMask
        for s = 1:nSeg
            itsc = masks(:, :, m) & segs(:, :, s);
            itsc = sum(itsc(:));
            if itsc > intsec(m)
                intsec(m) = itsc;
                u = masks(:, :, m) | segs(:, :, s);
                iou(m) = itsc / sum(u(:));
                match(:, :, m) = segs(:, :, s);
                originIdx(m) = i;
            end
        end
    end
end

% second 1 means top 1 here
match = reshape(match, [sz(1:2) 1 1 sz(3)]);
match = uint8(gather(repmat(match, [1 1 3 1 1])));

iou = gather(iou');


