addpath ../util/; Platform;
addpath ../dbcode/;

runStr = 'test3';

minSegSize = 256; % at 512x1024
sizes = [1024, 512, 256]; % for SegColor
outSize = [256 512]; % for SegList

classColor = uint8(dlmread('../dbcode/classColor.txt'));
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
    mask = cell(0, 1);
    classIdx = zeros(0, 1);
    for c = 1:nClass
        CC = bwconncomp(idMap == c);
        for s = 1:CC.NumObjects
            if numel(CC.PixelIdxList{s}) < minSegSize
                continue
            end
            m = false(CC.ImageSize);
            m(CC.PixelIdxList{s}) = true;
            m = imresize(m, outSize, 'nearest');
            mask{end+1} = m;
            classIdx(end+1) = c;
        end
    end
    save(fullfile(segListDir, [fileList{i}(1:end-3) 'mat']), 'mask', 'classIdx');
end





