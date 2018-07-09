ldata = load('CityscapesClasses.mat');
cc = ldata.CityscapesClasses;

classNames = strcat({cc.category}, {'.'}, {cc.name})';
colors = cell2mat({cc.color}');

hd = visclasscolor(classNames, colors, 'AllClass.png', false);