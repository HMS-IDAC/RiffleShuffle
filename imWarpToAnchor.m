function [M,F,tform] = imWarpToAnchor(i,anchor,Ks,tforms)

F = Ks{anchor};
T = [[eye(2) [0; 0]]; [0 0 1]];
if i < anchor   
    for j = i:anchor-1
        t = invert(tforms{j});
        T = T*t.T;
    end
    tform = affine2d(T);
    M = imwarp(Ks{i},tform,'OutputView',imref2d(size(F)));
elseif i > anchor
    T = [[eye(2) [0; 0]]; [0 0 1]];
    for j = i-1:-1:anchor
        t = tforms{j};
        T = T*t.T;
    end
    tform = affine2d(T);
    M = imwarp(Ks{i},tform,'OutputView',imref2d(size(F)));
else
    tform = affine2d([[eye(2) [0; 0]]; [0 0 1]]);
    M = F;
end

end