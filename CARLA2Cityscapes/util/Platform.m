% Setup host dependent code
[~, hostName] = system('hostname');
hostName = hostName(1:end-1);

switch hostName
    case 'mli-7720'
        dataDir = '/home/mli/Data';
        SSDDir = dataDir;
    case {'trinity.vision.cs.cmu.edu', ...
            'compute-0-11.local', ...
            'compute-0-12.local', ...
            'compute-0-14.local', ...
            'compute-0-15.local', ...
            'compute-0-17.local', ...
            'compute-0-20.local', ...
            'compute-0-21.local'}
        dataDir = '/data/mengtial';
        SSDDir = '/scratch/mengtial';
    case 'MT1080'
        dataDir = 'D:\Data';
        SSDDir = 'C:\Data';
    case 'mt1080ubt'
        dataDir = '/media/mt/ST2TB/Data';
        SSDDir = '/homde/mt/data';
    case 'MTSheep'
        dataDir = 'I:\Data';
        SSDDir = 'C:\Data';
    otherwise
        error('Please setup the correct path on this machine.');
end