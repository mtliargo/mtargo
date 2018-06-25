addpath util/; Platform;
addpath dbcode/;

runStr = 'test1';

minSegSize = 256; % at 512x1024
sizes = [1024, 512, 256]; % for SegColor
outSize = [256 512]; % for SegLIst

classColor = uint8(dlmread('dbcode/classColor.txt'));
nClass = size(classColor, 1);

seqDir = fullfile(dataDir, 'Exp/CARLA_gen17/e000001');
outDir = mkdir2(fullfile(dataDir, 'Exp/C2C', runStr));
segColorDir = fullfile(outDir, 'SegColor');
segColorOriginalSize = [segColorDir num2str(sizes(1), '-%d')];
copyfile(fullfile(seqDir, 'SegColor'), segColorOriginalSize);
segListDir = mkdir2(fullfile(outDir, ['SegList' num2str(outSize(1), '-%d')]));

fileList = dir(fullfile(segColorOriginalSize, '*.png'));
fileList = {fileList.name};

for i = 1:length(fileList)
    img = imread(fullfile(segColorOriginalSize, fileList{i}));
    for z = 2:length(sizes)
        outDirThis = mkdir2([segColorDir num2str(sizes(z), '-%d')]);
        imgResize = imresize(img, [sizes(z) 2*sizes(z)], 'nearest');
        imwrite(imgResize, fullfile(outDirThis, fileList{i}));
    end
    
    idMap = uint8(color2id(img, classColor));
    for c = 1:nClass
        outClassDir = mkdir2(fullfile(segListDir, num2str(c, '%02d')));
        CC = bwconncomp(idMap == c);
        masks = cell(0, 1);
        nMask = 0;
        for s = 1:CC.NumObjects
            if numel(CC.PixelIdxList{s}) < minSegSize
                continue
            end
            nMask = nMask + 1;
            masks{nMask} = zeros(CC.ImageSize, 'uint8');
            masks{nMask}(CC.PixelIdxList{s}) = 1;
            masks{nMask} = imresize(masks{nMask}, outSize, 'nearest');
        end
        
        masks = cat(3, masks{:});
        % SIMS format
        seg = struct;
        seg.library_mask = masks;
        seg.library_mask_pole = seg.library_mask;
        save(fullfile(outClassDir, [fileList{i}(1:end-3) 'mat']), '-struct', 'seg');
    end
end





