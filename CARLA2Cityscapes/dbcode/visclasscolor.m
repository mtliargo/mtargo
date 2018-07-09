function hd = visclasscolor(className, cmap, outName, bTransparent)

% the color values should be from 0-255

if nargin < 3
    outName = 'visclasscolor.png';
    bTransparent = true;
end

%%

wimg = 720;
himg = 1280;

s = 1;
nX = 1;
x0 = round(s*12);
y0 = round(s*12);
dx = round(s*430);
dy = round(s*100);
xbox = round(s*50);
ybox = round(s*50);
xbt = round(s*8);
yTextOffset = -8;
szFont = round(s*40);
font = 'LiberationSansNarrow-Regular';

%%

wimg = 1280;
himg = 380;

s = 0.52;
nX = 5;
x0 = round(s*10);
y0 = round(s*10);
dx = round(s*480);
dy = round(s*100);
xbox = round(s*50);
ybox = round(s*50);
xbt = round(s*10);
yTextOffset = -8;
szFont = round(s*40);
font = 'LiberationSansNarrow-Regular';

%%

canvas = 255*ones(himg, wimg, 3, 'uint8');
if bTransparent
    alphaCh = 255*ones(himg, wimg, 3, 'uint8');
end


ix = 1;
x = x0;
y = y0;

nClass = length(className);
hd = figure;

for i = 1:nClass
    canvas = insertShape(canvas, 'FilledRectangle', [x y xbox ybox], 'Color', cmap(i, :), 'Opacity', 1);
    if bTransparent
        alphaCh = insertShape(alphaCh, 'FilledRectangle', [x y xbox ybox], 'Color', [0 0 0], 'Opacity', 1);
    end
    canvas = insertText(canvas, [x + xbox + xbt, y + yTextOffset], className{i}, ...
         'FontSize', szFont, 'Font', font, 'BoxOpacity', 0);
    if bTransparent
        alphaCh = insertText(alphaCh, [x + xbox + xbt, y + yTextOffset], className{i}, ...
            'FontSize', szFont, 'Font', font, 'BoxOpacity', 0);
    end
     
    ix = ix + 1;
    if ix > nX
        ix = 1;
        x = x0;
        y = y + dy;
    else
        x = x + dx;
    end
end

imshow(canvas); truesize;
if bTransparent
    imwrite(canvas, outName, 'Alpha', im2double(255-alphaCh(:, :, 1)));
else
    imwrite(canvas, outName);
end


end