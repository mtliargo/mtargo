addpath ../util/; Platform;

minSegSize = 256; % at 512x1024

inDir = fullfile(dataDir, 'exp/CARLA_gen17/e000001/SegColor');
outDir = fullfile(dataDir, 'exp/CARLA_gen17/e000001/SegList');

fileList = dir(fullfile(inDir, '*.png'));
fileList = {fileList.name};

classColor = uint8(dlmread('classColor.txt'));
nClass = size(classColor, 1);

for i = 1:length(fileList)
   imgLabel = imread(fullfile(inDir, fileList{i}));
   idMap = uint8(color2id(imgLabel, classColor));
   
   for c = 1:nClass
       outClassDir = mkdir2(fullfile(outDir, num2str(c)));
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
       end
       
       masks = cat(3, masks{:});
       % SIMS format
       seg = struct;
       seg.library_mask = masks;
       seg.library_mask_pole = seg.library_mask;
       save(fullfile(outClassDir, [fileList{i}(1:end-3) 'mat']), '-struct', 'seg');
   end
end


