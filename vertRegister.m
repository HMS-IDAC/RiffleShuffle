function [tform,cs] = vertRegister(Moving,Fixed,varargin)

M = Moving;
F = Fixed;
M = M/sum(M(:));
F = F/sum(F(:));

ts = [];
cs = [];

tRange = -50:50;
if nargin > 2
    tRange = -varargin{1}:varargin{1};
end

for t = tRange
    TM = imtranslate(M,[0 t]);
    ts = [ts t];
    cs = [cs corr2(F,TM)];
end

[~,im] = max(cs);
t = ts(im);
% TMoving = imtranslate(Moving,[0 t]);

tform = affine2d([[eye(2) [0; 0]]; [0 t 1]]);

end