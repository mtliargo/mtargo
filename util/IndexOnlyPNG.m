Platform;

inDir = fullfile(dataDir, 'CarlaGen/C18_W2_S1/SegColor-1024');
outDir = mkdir2(replace(inDir, 'SegColor', 'Seg'));

splits = {'train', 'val', 'test'};

for s = 1:length(splits)
    split = splits{s};
    inDirSplit = fullfile(inDir, split);
    outDirSplit = mkdir2(fullfile(outDir, split));
    dirList = dir(fullfile(inDirSplit, '*.png'));
    parfor i = 1:length(dirList)
        [imap, cmap] = imread(fullfile(inDirSplit, dirList(i).name));
        imwrite(imap, fullfile(outDirSplit, dirList(i).name));
    end
end
