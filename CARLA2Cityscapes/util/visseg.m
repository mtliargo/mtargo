function visseg(fileList, classList, outDir, segListDir, matchDir, imgBankDir)

h = 256;
w = 2*h;

n = length(fileList);
nClass = length(classList);
for i = 1:n
    ch = zeros(h, w, 'uint8');
    chs = {ch, ch, ch};
    for c = 1:nClass
        classDir = fullfile(segListDir, classList{c});
        ldata = load(fullfile(classDir, fileList{i}));
        masks = logical(ldata.library_mask);

        if numel(masks) == 0
            continue;
        end
        
        matchClassDir = fullfile(matchDir, classList{c});
        ldata = load(fullfile(matchClassDir, fileList{i}));
        originIdx = ldata.response_index;

        for m = 1:size(masks, 3)
            trainImg = imread(fullfile(imgBankDir, num2str(originIdx(m), '%08d.png')));
            mask = masks(:, :, m);
            for channel = 1:3
                trainImgCh = trainImg(:, :, channel);
                chs{channel}(mask) = trainImgCh(mask);
            end
        end
    end
    img = cat(3, chs{:});
    imwrite(img, fullfile(outDir, [fileList{i}(1:end-3) 'png']));
end

end
