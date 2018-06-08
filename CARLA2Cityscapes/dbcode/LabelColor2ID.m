addpath ../util/; Platform;

workDir = fullfile(dataDir, 'Cityscapes/ReOrg');
labelDirs = dir(fullfile(workDir, 'label-color*'));
labelDirs = {labelDirs.name};

classColor = uint8(dlmread('classColor.txt'));
splits = {'train', 'val'};

parfor i = 1:length(labelDirs)
   inDir = fullfile(workDir, labelDirs{i});
   outDir = fullfile(workDir, strrep(labelDirs{i}, 'color', 'id'));
   
   for s = 1:length(splits)
       inDirThis = fullfile(inDir, splits{s});
       outDirThis = mkdir2(fullfile(outDir, splits{s}));
       
       fileList = dir(fullfile(inDirThis, '*.png'));
       fileList = {fileList.name};
       for j = 1:length(fileList)
           imgLabel = imread(fullfile(inDirThis, fileList{j}));
           idMap = uint8(color2id(imgLabel, classColor));
           imwrite(idMap, fullfile(outDirThis, fileList{j}));
       end
   end
end


