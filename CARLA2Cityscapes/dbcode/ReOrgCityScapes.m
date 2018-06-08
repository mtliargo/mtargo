addpath ../util/; Platform;

imgDir = fullfile(dataDir, 'Cityscapes/leftImg8bit');
labelDir = fullfile(dataDir, 'Cityscapes/gtFine');
outListDir = fullfile(dataDir, 'Cityscapes/ReOrg');
outImgDir = fullfile(dataDir, 'Cityscapes/ReOrg/image');
outLabelColorDir = fullfile(dataDir, 'Cityscapes/ReOrg/label-color');


splits = {'train', 'val', 'test'};
sizes = [1024, 512, 256, 128, 64];

parfor s = 1:length(splits)
    cities = dir(fullfile(imgDir, splits{s}));
    cities = {cities.name};
    cities = cities(~ismember(cities, {'.', '..'}));
    cities = sort(cities);
    
    idx = 0;
    fileList = [];
    
    for c = 1:length(cities)
        cityDir = fullfile(imgDir, splits{s}, cities{c});
        fileNames = dir(fullfile(cityDir, '*.png'));
        fileNames = arrayfun(@(x)(x.name(1:end-16)), fileNames, 'UniformOutput', false);
        fileNames = sort(fileNames);
        fileList = [fileList; fileNames];
        
        for i = 1:length(fileNames)
            idx = idx + 1;
            baseName = num2str(idx, '%08d');
            
            inDirThis = cityDir;
            outDirThis = mkdir2(fullfile([outImgDir num2str(sizes(1), '-%d')], splits{s}));
            inNameThis = fullfile(inDirThis, [fileNames{i} '_leftImg8bit.png']);
            img = imread(inNameThis);
            copyfile(inNameThis, fullfile(outDirThis, [baseName '.png']));
            
            if ~strcmp(splits{s}, 'test') 
                inDirThis = fullfile(labelDir, splits{s}, cities{c});
                outDirThis = mkdir2(fullfile([outLabelColorDir num2str(sizes(1), '-%d')], splits{s}));
                inNameThis = fullfile(inDirThis, [fileNames{i} '_gtFine_color.png']);
                imgLabel = imread(inNameThis);
                copyfile(inNameThis, fullfile(outDirThis, [baseName '.png']));
            end

            for z = 2:length(sizes)
                outDirThis = mkdir2(fullfile([outImgDir num2str(sizes(z), '-%d')], splits{s}));
                imgResize = imresize(img, [sizes(z) 2*sizes(z)]);
                imwrite(imgResize, fullfile(outDirThis, [baseName '.png']));

                if ~strcmp(splits{s}, 'test') 
                    imgLabelResize = imresize(imgLabel, [sizes(z) 2*sizes(z)], 'nearest');
                    outDirThis = mkdir2(fullfile([outLabelColorDir num2str(sizes(z), '-%d')], splits{s}));
                    imwrite(imgLabelResize, fullfile(outDirThis, [baseName '.png']));
                end
            end
        end
    end
    
    f = fopen(fullfile(outListDir, ['filename-' splits{s} '.txt']), 'w');
    fprintf(f, '%s\n', fileList{:});
    fclose(f);
end