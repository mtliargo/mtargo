function idMap = color2id(img, classColor)
%COLOR2ID Map colored label maps to class indices.
%    class indices are the trainId + 1 with other color being 0 (unlabeled) in 
%    https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

[w, h, ~] = size(img);
img = reshape(img, [], 3);
[~, idMap] = ismember(img, classColor, 'rows');

idMap = reshape(idMap, w, h);

end

