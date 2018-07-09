ldata = load('CityscapesClasses.mat');
cc = ldata.CityscapesClasses;

sel = ismember([cc.trainId], 0:18);
ccTrain = cc(sel);

classNames = strcat({ccTrain.category}, {'.'}, {ccTrain.name})';
colors = cell2mat({ccTrain.color}');

classNames = [classNames; 'void.ignored'];
colors = [colors; 0 0 0];

hd = visclasscolor(classNames, colors, 'TrainClass.png', false);