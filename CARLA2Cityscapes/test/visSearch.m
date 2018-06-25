function img = visSearch(mask, source, imgBankDir, height)
    
if nargin < 4
    height = 256;
end

h = height;
w = 2*h;

ch = zeros(h, w, 'uint8');
chs = {ch, ch, ch};

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
end

img = cat(3, chs{:});

end
