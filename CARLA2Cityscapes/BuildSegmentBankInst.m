addpath ../util/; Platform;
addpath dbcode/;

runStr = 'bank1';
minSegRatio = 1/32;
sizes = [256];
% sizes = [1024, 512, 256, 128, 64];

classColor = uint8(dlmread('dbcode/classColor.txt'));
nClass = size(classColor, 1);

inDir = fullfile(dataDir, 'Cityscapes/ReOrg');
outDir = mkdir2(fullfile(dataDir, 'Exp/C2C', runStr));

for s = 1:length(sizes)
    sizeStr = num2str(sizes(s));
    minSegSize = sizes(s)*minSegRatio;
    outSize = [sizes(s) 2*sizes(s)];
    
    segColorDir = fullfile(inDir, ['SegColor-' sizeStr], 'train');
    fileList = dir(fullfile(segColorDir, '*.png'));
    fileList = {fileList.name};

    classAllMask = cell(nClass, 1);
    classAllSource = cell(nClass, 1);
    for c = 1:nClass
        classAllMask{c} = cell(0, 1);
        classAllSource{c} = zeros(0, 1);
    end
    for i = 1:length(fileList)
        img = imread(fullfile(segColorDir, fileList{i}));
        idMap = uint8(color2id(img, classColor));
        for c = 1:nClass
            CC = bwconncomp(idMap == c);
            for j = 1:CC.NumObjects
                if numel(CC.PixelIdxList{j}) < minSegSize
                    continue
                end
                m = false(CC.ImageSize);
                m(CC.PixelIdxList{j}) = 1;
                m = imresize(m, outSize, 'nearest');
                classAllMask{c}{end+1} = m;
                classAllSource{c}(end+1) = i;
            end
        end
    end
    for c = 1:nClass
        mask = classAllMask{c};
        source = classAllSource{c};
        save(fullfile(outDir, num2str(c, '%02d.mat')), '-v7.3', 'mask', 'source');
    end
end









