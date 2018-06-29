function [img, orgImgs] = visSearch(mask, classIdx, match, source, imgBankDir, height)
    
if nargin < 6
    height = 256;
end

h = height;
w = 2*h;

ch = zeros(h, w, 'uint8');
chs = {ch, ch, ch};

nRoad = 0;
nCar = 0; 
nBuilding = 0;
orgImgs = cell(5, 1);
orgMaskColor = [15, 255, 247];
newMaskColor = [229, 244, 66];
overlayOpacity = 0.5;

list = find(source ~= 0);
n = length(list);
for i = 1:n
    idx = list(i);
    trainImg = imread(fullfile(imgBankDir, num2str(source(idx), '%08d.png')));
    m = mask{idx};
    for channel = 1:3
        trainImgCh = trainImg(:, :, channel);
        chs{channel}(m) = trainImgCh(m);
    end
    
    m2 = match{idx};
    if nRoad < 1 && classIdx(idx) == 1
        nRoad = nRoad + 1;
        outChs = cell(3, 1);
        for channel = 1:3
            trainImgCh = trainImg(:, :, channel);
            trainImgCh(m2) = (1-overlayOpacity)*trainImgCh(m2) + overlayOpacity*orgMaskColor(channel);
            trainImgCh(m) = (1-overlayOpacity)*trainImgCh(m) + overlayOpacity*newMaskColor(channel);
            outChs{channel} = trainImgCh;
        end
        orgImgs{nRoad} = cat(3, outChs{:});
    end
    
    if nCar < 3 && classIdx(idx) == 14
        nCar = nCar + 1;
        outChs = cell(3, 1);
        for channel = 1:3
            trainImgCh = trainImg(:, :, channel);
            trainImgCh(m2) = (1-overlayOpacity)*trainImgCh(m2) + overlayOpacity*orgMaskColor(channel);
            trainImgCh(m) = (1-overlayOpacity)*trainImgCh(m) + overlayOpacity*newMaskColor(channel);
            outChs{channel} = trainImgCh;
        end
        orgImgs{nCar + 1} = cat(3, outChs{:});
    end
    
    if nBuilding < 1 && classIdx(idx) == 3
        nBuilding = nBuilding + 1;
        outChs = cell(3, 1);
        for channel = 1:3
            trainImgCh = trainImg(:, :, channel);
            trainImgCh(m2) = (1-overlayOpacity)*trainImgCh(m2) + overlayOpacity*orgMaskColor(channel);
            trainImgCh(m) = (1-overlayOpacity)*trainImgCh(m) + overlayOpacity*newMaskColor(channel);
            outChs{channel} = trainImgCh;
        end
        orgImgs{nBuilding + 4} = cat(3, outChs{:});
    end
end

img = cat(3, chs{:});

end
