addpath ../../util/; Platform;

imgDir = fullfile(dataDir, 'Cityscapes/leftImg8bit');
labelDir = fullfile(dataDir, 'Cityscapes/gtFine');
outListDir = fullfile(dataDir, 'Cityscapes/ReOrg');
outImgDir = fullfile(dataDir, 'Cityscapes/ReOrg/Image');
outLabelColor35Dir = fullfile(dataDir, 'Cityscapes/ReOrg/SegColor35');
outLabelColor20Dir = fullfile(dataDir, 'Cityscapes/ReOrg/SegColor20');

splits = {'train', 'val', 'test'};
sizes = [1024, 512, 256, 128, 64];

ldata = load('CityscapesClasses.mat');
cc = ldata.CityscapesClasses;

colors35 = double(cell2mat({cc.color}'))/255;

sel = ismember([cc.trainId], 0:18);
ccTrain = cc(sel);
colors20 = cell2mat({ccTrain.color}');
colors20 = double([colors20; 0 0 0])/255;

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
                outDirThis = mkdir2(fullfile([outLabelColor35Dir num2str(sizes(1), '-%d')], splits{s}));
                inNameThis = fullfile(inDirThis, [fileNames{i} '_gtFine_labelIds.png']);
                imgLabel35 = imread(inNameThis);
                imwrite(imgLabel35, colors35, fullfile(outDirThis, [baseName '.png']));   
                
                inDirThis = fullfile(labelDir, splits{s}, cities{c});
                outDirThis = mkdir2(fullfile([outLabelColor20Dir num2str(sizes(1), '-%d')], splits{s}));
                inNameThis = fullfile(inDirThis, [fileNames{i} '_gtFine_labelTrainIds.png']);
                imgLabel20 = imread(inNameThis);
                imgLabel20(imgLabel20 == 255) = 19;
                imwrite(imgLabel20, colors20, fullfile(outDirThis, [baseName '.png']));             
            end

            for z = 2:length(sizes)
                outDirThis = mkdir2(fullfile([outImgDir num2str(sizes(z), '-%d')], splits{s}));
                imgResize = imresize(img, [sizes(z) 2*sizes(z)]);
                imwrite(imgResize, fullfile(outDirThis, [baseName '.png']));

                if ~strcmp(splits{s}, 'test') 
                    imgLabel35Resize = imresize(imgLabel35, [sizes(z) 2*sizes(z)], 'nearest');
                    outDirThis = mkdir2(fullfile([outLabelColor35Dir num2str(sizes(z), '-%d')], splits{s}));
                    imwrite(imgLabel35Resize, colors35, fullfile(outDirThis, [baseName '.png']));
                    
                    imgLabel20Resize = imresize(imgLabel20, [sizes(z) 2*sizes(z)], 'nearest');
                    outDirThis = mkdir2(fullfile([outLabelColor20Dir num2str(sizes(z), '-%d')], splits{s}));
                    imwrite(imgLabel20Resize, colors20, fullfile(outDirThis, [baseName '.png']));                   
                end
            end
        end
    end
    
    f = fopen(fullfile(outListDir, ['filename-' splits{s} '.txt']), 'w');
    fprintf(f, '%s\n', fileList{:});
    fclose(f);
end