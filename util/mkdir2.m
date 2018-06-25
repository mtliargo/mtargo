function pathName = mkdir2(pathName)

if ~exist(pathName, 'dir')
    mkdir(pathName);
end

end