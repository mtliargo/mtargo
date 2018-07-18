Platform;

inDir = fullfile(dataDir, 'Cityscapes/ReOrg/Image-512');
outDir = mkdir2([inDir '-JPG']);

splits = {'train', 'val', 'test'};

for s = 1:length(splits)
    split = splits{s};
    inDirSplit = fullfile(inDir, split);
    outDirSplit = mkdir2(fullfile(outDir, split));
    dirList = dir(fullfile(inDirSplit, '*.png'));
    for i = 1:length(dirList)
        img = imread(fullfile(inDirSplit, dirList(i).name));
        imwrite(img, fullfile(outDirSplit, [dirList(i).name(1:end-3) 'jpg']));
    end
end
