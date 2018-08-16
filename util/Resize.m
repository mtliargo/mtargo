Platform;

ifSeg = 1;
if ifSeg
    inDir = fullfile(dataDir, 'CarlaGen/C20_S1/Seg-1024');
else
    inDir = fullfile(dataDir, 'CarlaGen/C20_S1/RGB-1024');
end
outDir = mkdir2([inDir(1:end-4) '512']);

splits = {'train', 'val', 'test'};

for s = 1:length(splits)
    split = splits{s};
    inDirSplit = fullfile(inDir, split);
    outDirSplit = mkdir2(fullfile(outDir, split));
    dirList = dir(fullfile(inDirSplit, '*.png'));
    parfor i = 1:length(dirList)
        if ifSeg
            [idx, cm] = imread(fullfile(inDirSplit, dirList(i).name));
            idx = imresize(idx, [512 1024], 'nearest');
            imwrite(idx, cm, fullfile(outDirSplit, dirList(i).name));
        else
            img = imread(fullfile(inDirSplit, dirList(i).name));
            img = imresize(img, [512 1024]);
            imwrite(img, fullfile(outDirSplit, dirList(i).name));
        end
    end
end
