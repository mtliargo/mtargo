function parsave(filename, varstruct, varargin)
%PARSAVE Save worksapce variables to file in a parfor context.
%   PARSAVE(FILENAME, VARSTRUCT) stores variables packed in 
%   VARSTRUCT to a MAT-file named FILENAME.
%
%   PARSAVE(FILENAME, VARSTRUCT, OPTIONS) specifiy additional options
%   interpreted by the SAVE function (e.g. '-v7.3', '-nocompression').
%
%   Due to the transparency limitation of parfor loops, all the variables
%   to be stroed must be packed into a scalar structure variable
%   (VARSTRUCT) as input argument. This is intended to capture both the 
%   name and the value of the variables.
%
%   Examples:
%     parfor i = 1:10
%         a = 2*i+1;
%         b = 3*i-1;
%
%         st = struct;
%         st.a = a;
%         st.b = b;
%         parsave(num2str(i, '%02d.mat'), st, '-v7.3');
%     end
%
%   See also SAVE.

%   by Martin Li, 2018

if ~isstruct(varstruct)
    error('The input variables must be packed into a scalar structure variable.');
end

save(filename, '-struct', 'varstruct', varargin{:});
