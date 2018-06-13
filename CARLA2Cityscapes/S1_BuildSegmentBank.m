% using SIMS' seg bank for now

% addpath util/; Platform;
% 
% searchRes = 256;
% fullRes = 1024;
% 
% minSegSize = 9; % in the search resolution
% 
% inDir = fullfile(dataDir, 'Cityscapes/ReOrg');
% outDir = mkdir2(fullfile(dataDir, 'Exp/C2C/SegBank'));
% 
% searchResDir = fullfile(inDir, num2str(searchRes, 'label-id-%d'));
% fullResImgDir = fullfile(inDir, num2str(fullRes, 'image-%d')); 
% fullResLabelDir = fullfile(inDir, num2str(fullRes, 'label-id-%d'));
% 
% fileList = dir(fullfile(searchResDir, 'train/*.png'));
% fileList = {fileList.name};
% fileList = sort(fileList);
% 
% classColor = uint8(dlmread('dbcode/classColor.txt'));
% nClass = size(classColor, 1);
% 
% for i = 1:length(fileList)
%     bwconncomp
%     
% end


% amodel road with depth ordering





