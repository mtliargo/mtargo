mx = 2e-6; % meters per pixel

x_res = 2048;

K = [2262.52 0 1096.98; 0 2265.3017905988554 513.137; 0 0 1];


a = rad2deg(atan2(K(1, 3), K(1, 1)))
b = rad2deg(atan2(x_res - K(1, 3), K(1, 1)))
fov = a + b
