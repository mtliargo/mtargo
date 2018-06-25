# Setup host dependent code
from platform import node
hostname = node()

if hostname == 'mli-7720':
    data_dir = '/home/mli/Data'
    SSD_dir = data_dir
elif hostname == 'MT1080':
    data_dir = r'D:\Data'
    SSD_dir = r'C:\Data'
else:
    raise Exception('Please setup the correct path on this machine.')
