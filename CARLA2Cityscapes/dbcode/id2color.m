function img = id2color(idMap, classColor)
%ID2COLOR Map class indices to to RGB colors.
%    class indices are the trainId + 1 with other color being 0 (unlabeled) in 
%    https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

classColor = [0 0 0; classColor];
idMap = idMap + 1;

img = classColor(idMap(:), :);
img = reshape(idMap, [size(idMap) 3]);

end

