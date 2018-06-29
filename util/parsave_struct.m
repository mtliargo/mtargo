function parsave(filename, varstruct, varargin)
%PARSAVE Save variables in a parfor context.
%   PARSAVE(FILENAME, VARSTRUCT [, FORMATSTRS]) saves variables packed in 
%   VARSTRUCT to FILENAME. The FORMATSTRS specify additional options used
%   in the SAVE function (e.g. '-v7.3').
%
%   Due to the transparency limitation of parfor loops, all the input
%   variables must be packed into a scalar sutructure variable (VARSTRUCT) 
%   as an argument to to this function.
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
%
%   See also SAVE.
%
%   by Martin Li, 2018

if ~isstruct(varstruct)
    error('The input variables must be packed into a scalar structure variable.');
end

save(filename, '-struct', 'varstruct', varargin{:});
