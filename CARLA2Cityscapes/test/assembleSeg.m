function canvas = assembleSeg(mask, classIdx, source, imgBankDir, idMap, height)
% Assemble the searched segments with some processing for the GAN

% Processing:
% General elision
% Bottom elision for some classes
% Special handling for certain classes

if nargin < 5
    height = 512;
end

h = height;
w = 2*h;

%% Inner elision
elisionRatio = 0.5;
se = strel('square', floor(0.035*h));

%% Outer elision
% 0 - invalid, 1 - road, 2 - sidewalk
maskContext = idMap == 0 | idMap == 1 | idMap == 2;
objectClass = [3,6,7,8,12,13,14,15,16,17,18,19];
seOuter = strel('square', floor(0.08*h));
extRatio = 0.125;

ch = zeros(h, w, 'uint8');
chs = {ch, ch, ch};

list = find(source ~= 0);
n = length(list);
for i = 1:n
    idx = list(i);
    trainImg = imread(fullfile(imgBankDir, num2str(source(idx), '%08d.png')));
    m = mask{idx};
    if ~isequal(size(m), [h w])
        m = imresize(m, [h w], 'nearest');
    end
    
    %% Inner elision
    bd = edge(m, 'Sobel', 0);
    bdDilate = imdilate(single(bd), se);
    % thick black boundary
    bdDilate = bdDilate==1;
    bdIdx = find(bdDilate);
    nBd = length(bdIdx);
    % white dots index (value 255)
    bdIdx = bdIdx(randperm(nBd, floor(elisionRatio*nBd)));
    
    %% Outer elision
    bOuterElision = ismember(classIdx(idx), objectClass);
    bdDilateOuter = imdilate(single(bd), seOuter);
    [rows, cols] = find(m);
    rmin = min(rows);
    rmax = max(rows);
    cmin = min(cols);
    cmax = max(cols);
    mh = rmax - rmin + 1;
    mw = cmax - rmin + 1;
    rstart = rmin;
    cstart = max(1, round(cmin - mw*extRatio));
    rend = min(h, round(rmax + mh*extRatio));
    cend = min(w, round(rmax + mw*extRatio));
    mbox = false(h, w);
    mbox(rstart:rend, cstart:cend) = 1;
    
    outerElisionMask = maskContext & mbox & bdDilateOuter;
    
    for channel = 1:3
        trainImgCh = trainImg(:, :, channel);
        chs{channel}(m) = trainImgCh(m);
        chs{channel}(bdDilate) = 0;
        chs{channel}(bdIdx) = 255;
        if bOuterElision
            chs{channel}(outerElisionMask) = 0;
        end
    end
    
    
end

canvas = cat(3, chs{:});

end
