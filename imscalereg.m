function [TMoving,c2,tform] = imscalereg(Moving,Fixed,rxs,rys,dys)

I = Moving/sum(Moving(:));
J = Fixed/sum(Fixed(:));

% rxs = 0.5:0.1:1.5;
% rys = 0.5:0.1:1.5;
% dys = -10:10;

rs = zeros(length(rxs)*length(rys)*length(dys),4);
count = 0;
for i = 1:length(rxs)
    for j = 1:length(rys)
        for k = 1:length(dys)
            count = count+1;
            rs(count,1:3) = [rxs(i) rys(j) dys(k)];
        end
    end
end

tforms = cell(1,size(rs,1));
for i = 1:size(rs,1)
    mp = size(I)/2;
    T1 = [eye(2) [0; 0]; [-mp(2) -mp(1) 1]]';
    rx = rs(i,1); ry = rs(i,2); dy = rs(i,3);
    r = [rx 0; 0 ry];
    R = [[r [0; 0]]; [0 dy 1]]';
    mp = size(J)/2;
    T2 = [eye(2) [0; 0]; [mp(2) mp(1) 1]]';
    tform = affine2d((T2*R*T1)');
    tforms{i} = tform;
    TI = imwarp(I,tform,'OutputView',imref2d(size(J)));
    rs(i,4) = (corr2(TI,J)+1)/2;
end

[c2,im] = max(rs(:,4));
tform = tforms{im};
TMoving = imwarp(I,tform,'OutputView',imref2d(size(J)));

end