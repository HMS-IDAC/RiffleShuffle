%% setup

clear, clc

%\\\SET
    interactive = true;
    % true: interactive mode (run cell by cell)
    % false: fast track (run by calling rsStackRegistration in Matlab's prompt)

    quantSpots = true;
     % true: quantify spots
     % false: quantify diffuse signal

    pathIn = '/scratch/RiffleShuffle/Stacks/Dataset_B_1.55_2.45_Downsized';
    % full path to _Downsized folder
%///


pathOut = [pathIn '_Registered'];

if ~exist(pathOut,'dir')
    mkdir(pathOut);
end

nImages = length(listfiles(pathIn,'C1.tif'));
padFactor = 1.5;

%% [interactive] automated vertical symmetry registration

if interactive
lI0 = cell(1,nImages);
lI1 = cell(1,nImages);
lI2 = cell(1,nImages);
lIQ = cell(1,nImages);
lTform = cell(1,nImages);
spots = cell(1,nImages);
xyOffsets = cell(1,nImages);
pfpb = pfpbStart(nImages);
parfor i = 1:nImages
    I = imreadGrayscaleDouble([pathIn filesep sprintf('I%03d_C.png',i)]);
    [I,xOffset] = padLR(I,padFactor);
    [I,yOffset] = padTB(I,padFactor);
    
    I1 = I;
    
    if quantSpots
        xy = fliplr(table2array(readtable([pathIn filesep sprintf('I%03d.csv',i)])));
    else
        xy = [1 1]; % dummy array
    end
    xy(:,1) = xy(:,1)+xOffset;
    xy(:,2) = xy(:,2)+yOffset;
    spots{i} = xy;
    xyOffsets{i} = [xOffset yOffset];
    
    I = imerode(I,strel('disk',2,0));
    I = I.*(I > 0.5*max(I(:)));
    I = normalize(steerableDetector(im2double(I),4,10));
    f = 200/max(size(I));
    I = fadeLR(imresize(I,f),0.25);

    [angles, midPoints] = symmetryViaRegistration2D(I,'RegMethod','nxc','AngleSet',-30:6:30);
    ag = angles(1);
    if ag > pi/2
        ag = ag-pi;
    end
    mp = midPoints(:,1);
    d = sqrt(size(I,1)^2+size(I,2)^2);
    v = [cos(ag); sin(ag)];
    ps = [];
    for j = -d:d
        p = round(mp+j*v);
        if p(1) >= 1 && p(1) <= size(I,1) && p(2) >= 1 && p(2) <= size(I,2)
            ps = [ps; p'];
        end
    end
    mp = mean(ps);
    
    T0 = [f*eye(2) [0; 0]; [0 0 1]]';
    T1 = [eye(2) [0; 0]; [-mp(2) -mp(1) 1]]';
    r = [cos(-ag) -sin(-ag); sin(-ag) cos(-ag)];
    R = [[r [0; 0]]; [0 0 1]]';
    T2 = [eye(2) [0; 0]; [mp(2) mp(1) 1]]';
    T3 = [eye(2) [0; 0]; [size(I,2)/2-mp(2) size(I,1)/2-mp(1) 1]]';
    T4 = [1/f*eye(2) [0; 0]; [0 0 1]]';
    tform = affine2d((T4*T3*T2*R*T1*T0)');
    
    TI1 = imwarp(I1,tform,'OutputView',imref2d(size(I1)));
    
    I0 = imreadGrayscaleDouble([pathIn filesep sprintf('I%03d_C1.tif',i)]);
    I0 = padTB(padLR(I0,padFactor),padFactor);
    TI0 = imwarp(I0,tform,'OutputView',imref2d(size(I1)));
    
    I2 = imreadGrayscaleDouble([pathIn filesep sprintf('I%03d_M.png',i)]);
    I2 = padTB(padLR(I2,padFactor),padFactor);
    TI2 = imwarp(I2,tform,'OutputView',imref2d(size(I1)));
    
    IQ = imreadGrayscaleDouble([pathIn filesep sprintf('I%03d_CQ.tif',i)]);
    IQ = padTB(padLR(IQ,padFactor),padFactor);
    TIQ = imwarp(IQ,tform,'OutputView',imref2d(size(I1)));
    
    lI0{i} = TI0;
    lI1{i} = TI1;
    lI2{i} = TI2;
    lIQ{i} = TIQ;
    lTform{i} = tform;

    pfpbUpdate(pfpb);
end

%  save variables to later skip step above
save([pathOut filesep 'lTform.mat'],'lTform');
save([pathOut filesep 'xyOffsets.mat'],'xyOffsets');
disp('done with symmetry detection')
end

%% [fast track] automated vertical symmetry registration

disp('vertical symmetry registration')
load([pathOut filesep 'lTform.mat']);
load([pathOut filesep 'xyOffsets.mat']);
lI0 = cell(1,nImages);
lI1 = cell(1,nImages);
lI2 = cell(1,nImages);
lIQ = cell(1,nImages);
spots = cell(1,nImages);
for i = 1:nImages
    disp(i)

    tform = lTform{i};
    
    I1 = imreadGrayscaleDouble([pathIn filesep sprintf('I%03d_C.png',i)]);
    I1 = padTB(padLR(I1,padFactor),padFactor);
    lI1{i} = imwarp(I1,tform,'OutputView',imref2d(size(I1)));
    
    I0 = imreadGrayscaleDouble([pathIn filesep sprintf('I%03d_C1.tif',i)]);
    I0 = padTB(padLR(I0,padFactor),padFactor);
    lI0{i} = imwarp(I0,tform,'OutputView',imref2d(size(I1)));
    
    I2 = imreadGrayscaleDouble([pathIn filesep sprintf('I%03d_M.png',i)]);
    I2 = padTB(padLR(I2,padFactor),padFactor);
    lI2{i} = imwarp(I2,tform,'OutputView',imref2d(size(I1)));
    
    IQ = imreadGrayscaleDouble([pathIn filesep sprintf('I%03d_CQ.tif',i)]);
    IQ = padTB(padLR(IQ,padFactor),padFactor);
    lIQ{i} = imwarp(IQ,tform,'OutputView',imref2d(size(I1)));
    
    if quantSpots
        xy = fliplr(table2array(readtable([pathIn filesep sprintf('I%03d.csv',i)])));
    else
        xy = [1 1]; % dummy array
    end
    xy(:,1) = xy(:,1)+xyOffsets{i}(1);
    xy(:,2) = xy(:,2)+xyOffsets{i}(2);
    spots{i} = xy;
end

%% transform spots

disp('transform spots')
spots2 = cell(1,nImages);
for i = 1:nImages
    disp(i)
    xy = transformSpots(spots{i},lTform{i});
    spots2{i} = xy;
end

%% create stacks

disp('create stacks')
szs = zeros(nImages,2);
for i = 1:nImages
    szs(i,:) = size(lI1{i});
end

s12 = max(szs)+10;
S0 = zeros(s12(1),s12(2),nImages);
S1 = zeros(s12(1),s12(2),nImages);
S2 = zeros(s12(1),s12(2),nImages);
SQ = zeros(s12(1),s12(2),nImages);

for i = 1:nImages
    disp(i)
    r0 = round(s12(1)/2-size(lI0{i},1)/2);
    c0 = round(s12(2)/2-size(lI0{i},2)/2);
    S0(r0+1:r0+size(lI0{i},1),c0+1:c0+size(lI0{i},2),i) = lI0{i};
    S1(r0+1:r0+size(lI0{i},1),c0+1:c0+size(lI0{i},2),i) = lI1{i};
    S2(r0+1:r0+size(lI0{i},1),c0+1:c0+size(lI0{i},2),i) = lI2{i};
    SQ(r0+1:r0+size(lI0{i},1),c0+1:c0+size(lI0{i},2),i) = lIQ{i};
    xy = spots2{i};
    xy(:,1) = xy(:,1)+c0+1;
    xy(:,2) = xy(:,2)+r0+1;
    spots2{i} = xy;
end

%% check if spots were transformed correctly

if interactive
for i = 1:nImages
    disp(i)
    I = imadjust(SQ(:,:,i));
    xy = spots2{i};
    imshow(I), hold on
    plot(xy(:,1),xy(:,2),'o'), hold off
%     pause
    pause(0.5)
end
close all
end

%% visualize (S0: C1, S1: C, S2: M, SQ: Q)

if interactive
timeLapseViewTool(S1);
end

%% [interactive] check/adjust symmetry (do not execute more than once)
% run with imIndices = [] if no adjustment is needed (to setup spots3 variable)

if interactive
maxShift = 500;
maxSink = 100;

%\\\SET
    % indices of images that need the axis of symmetry to be fixed; multiple indices separated by space
    % run with imIndices = [] if no adjustment is needed (to setup spots3 variable)
    imIndices = [2 3 10];
%///

spots3 = cell(1,nImages);
tforms23 = cell(1,nImages);
for i = 1:nImages
    spots3{i} = spots2{i};
    tforms23{i} = [];
end
for i = imIndices
    disp(i)
    T = symmetryTool(imadjust(S0(:,:,i)),'MaxShift',maxShift,'MaxSink',maxSink);
    if T.DoneButtonPushed
        S0(:,:,i) = imwarp(S0(:,:,i),T.Tform,'OutputView',imref2d(size(S0(:,:,i))));
        S1(:,:,i) = imwarp(S1(:,:,i),T.Tform,'OutputView',imref2d(size(S1(:,:,i))));
        S2(:,:,i) = imwarp(S2(:,:,i),T.Tform,'OutputView',imref2d(size(S2(:,:,i))));
        SQ(:,:,i) = imwarp(SQ(:,:,i),T.Tform,'OutputView',imref2d(size(SQ(:,:,i))));
        spots3{i} = transformSpots(spots2{i},T.Tform);
        tforms23{i} = T.Tform;
    end
end
save([pathOut filesep 'tforms23.mat'],'tforms23');
end

%% [fast track] check/adjust symmetry (do not execute more than once)

disp('adjust symmetry')
load([pathOut filesep 'tforms23.mat'])
imIndices = [];
spots3 = cell(1,nImages);
for i = 1:nImages
    disp(i)
    tform = tforms23{i};
    spots3{i} = spots2{i};
    if ~isempty(tform)
        S0(:,:,i) = imwarp(S0(:,:,i),tform,'OutputView',imref2d(size(S0(:,:,i))));
        S1(:,:,i) = imwarp(S1(:,:,i),tform,'OutputView',imref2d(size(S1(:,:,i))));
        S2(:,:,i) = imwarp(S2(:,:,i),tform,'OutputView',imref2d(size(S2(:,:,i))));
        SQ(:,:,i) = imwarp(SQ(:,:,i),tform,'OutputView',imref2d(size(SQ(:,:,i))));
        spots3{i} = transformSpots(spots2{i},tform);
        imIndices = [imIndices i];
    end
end

%% check if spots were transformed correctly

if interactive
for i = imIndices
    disp(i)
    I = imadjust(SQ(:,:,i));
    xy = spots3{i};
    imshow(I), hold on
    plot(xy(:,1),xy(:,2),'o'), hold off
    pause
end
close all
end

%% visualize (S0: C1, S1: C, S2: M, SQ: Q)

if interactive
timeLapseViewTool(S1);
end

%% [interactive] check/adjust symmetry on each left or right half (do not execute more than once)
% run with imIndices = [] if no adjustment is needed (to setup spots4 variable)

if interactive
maxShift = 500;
maxSink = 100;

%\\\SET
    % indices of images that need each half to be fixed independently; multiple indices separated by space
    % run with imIndices = [] if no adjustment is needed (to setup spots4 variable)
    imIndices = [3];
%///

spots4 = cell(1,nImages);
tforms34 = cell(1,nImages);
for i = 1:nImages
    spots4{i} = spots3{i};
    tforms34{i} = [];
end
for i = imIndices
    disp(i)
    T = symmetryTool2(imadjust(S0(:,:,i)),'MaxShift',maxShift,'MaxSink',maxSink);
    if T.DoneButtonPushed
        S0(:,:,i) = symmetryTool2.staticApplyTforms(T.MidC,T.Tform{1},T.Tform{2},S0(:,:,i));
        S1(:,:,i) = symmetryTool2.staticApplyTforms(T.MidC,T.Tform{1},T.Tform{2},S1(:,:,i));
        S2(:,:,i) = symmetryTool2.staticApplyTforms(T.MidC,T.Tform{1},T.Tform{2},S2(:,:,i));
        SQ(:,:,i) = symmetryTool2.staticApplyTforms(T.MidC,T.Tform{1},T.Tform{2},SQ(:,:,i));
        
        spots4{i} = symmetryTool2.staticApplyTformsToSpots(T.MidC,T.Tform{1},T.Tform{2},spots3{i});
        tforms34{i} = {T.MidC, T.Tform{1}, T.Tform{2}};
    end
end
save([pathOut filesep 'tforms34.mat'],'tforms34');
end

%% [fast track] check/adjust symmetry on each left or right half (do not execute more than once)

disp('adjust symmetry on each half')
load([pathOut filesep 'tforms34.mat']);
spots4 = cell(1,nImages);
imIndices = [];
for i = 1:nImages
    disp(i)
    tform = tforms34{i};
    spots4{i} = spots3{i};
    if ~isempty(tform)
        S0(:,:,i) = symmetryTool2.staticApplyTforms(tform{1},tform{2},tform{3},S0(:,:,i));
        S1(:,:,i) = symmetryTool2.staticApplyTforms(tform{1},tform{2},tform{3},S1(:,:,i));
        S2(:,:,i) = symmetryTool2.staticApplyTforms(tform{1},tform{2},tform{3},S2(:,:,i));
        SQ(:,:,i) = symmetryTool2.staticApplyTforms(tform{1},tform{2},tform{3},SQ(:,:,i));
        spots4{i} = symmetryTool2.staticApplyTformsToSpots(tform{1},tform{2},tform{3},spots3{i});
        imIndices = [imIndices i];
    end
end

%% check if spots were transformed correctly

if interactive
for i = imIndices
    disp(i)
    I = imadjust(SQ(:,:,i));
    xy = spots4{i};
    imshow(I), hold on
    plot(xy(:,1),xy(:,2),'o'), hold off
    pause
end
close all
end

%% [interactive] pairwise vertical registration

if interactive
sI0 = cell(1,nImages);
sI1 = cell(1,nImages);
sI2 = cell(1,nImages);
sIQ = cell(1,nImages);
rI1 = cell(1,nImages);
for i = 1:nImages
    disp(i)
    sI0{i} = S0(:,:,i);
    sI1{i} = S1(:,:,i);
    sI2{i} = S2(:,:,i);
    sIQ{i} = SQ(:,:,i);
    rI1{i} = imresize(S1(:,:,i),0.1);
end

tforms = cell(1,nImages-1);
for i = 1:nImages-1
    disp(i)
    I0 = rI1{i};
    I1 = rI1{i+1};
    [tform,cs] = vertRegister(I1,I0,10);
    tform0 = tform;
    tform.T(3,2) = 10*tform.T(3,2);
	tforms{i} = tform;
    imshowpair(I0,imwarp0(I1,tform0))
    pause(0.1)
end
close all
save([pathOut filesep 'tforms.mat'],'tforms');
end

%% [fast track] pairwise vertical registration

disp('pairwise vertical registration')
load([pathOut filesep 'tforms.mat']);
sI0 = cell(1,nImages);
sI1 = cell(1,nImages);
sI2 = cell(1,nImages);
sIQ = cell(1,nImages);
rI1 = cell(1,nImages);
for i = 1:nImages
    disp(i)
    sI0{i} = S0(:,:,i);
    sI1{i} = S1(:,:,i);
    sI2{i} = S2(:,:,i);
    sIQ{i} = SQ(:,:,i);
    rI1{i} = imresize(S1(:,:,i),0.1);
end

%% global vertical registration

disp('global vertical registration')
anchor = round(nImages/2);
TS0 = zeros(size(S0));
TS1 = zeros(size(S1));
TS2 = zeros(size(S2));
TSQ = zeros(size(SQ));

spots5 = cell(1,nImages);
for i = 1:nImages
    disp(i)
    [M,~,tform] = imWarpToAnchor(i,anchor,sI0,tforms);
    TS0(:,:,i) = M;
    if exist('spots4','var')
        spots5{i} = transformSpots(spots4{i},tform);
    elseif exist('spots3','var')
        spots5{i} = transformSpots(spots3{i},tform);
    else
        spots5{i} = transformSpots(spots2{i},tform);
    end
    M = imWarpToAnchor(i,anchor,sI1,tforms);
    TS1(:,:,i) = M;
    M = imWarpToAnchor(i,anchor,sI2,tforms);
    TS2(:,:,i) = M;
    M = imWarpToAnchor(i,anchor,sIQ,tforms);
    TSQ(:,:,i) = M;
end


%% check if spots were transformed correctly

if interactive
for i = 1:nImages
    disp(i)
    I = imadjust(TSQ(:,:,i));
    xy = spots5{i};
    imshow(I), hold on
    plot(xy(:,1),xy(:,2),'o'), hold off
    pause(0.5)
end
close all
end

%% visualize (S0: C1, S1: C, S2: M, SQ: Q)

if interactive
timeLapseViewTool(TS1);
end

%% visualize all channels per index

if interactive
    
%\\\SET
    % index of plane to visualize
    index = 3;
%///

stack = imadjust(TS0(:,:,index));
stack = cat(3,stack,TS1(:,:,index));
stack = cat(3,stack,TS2(:,:,index));
stack = cat(3,stack,imadjust(TSQ(:,:,index)));
tlvt(stack);
end

%% [interactive] adjust (if needed; when done, re-do 'global vertical registration')

if interactive

%\\\SET
    imIndices = [10]; % indices of images to adjust;
    % for each index, adjusts corresponding image w.r.t. previous image
%///

for i = imIndices
    T = verticalRegistrationTool(imadjust(S1(:,:,i)),imadjust(S1(:,:,i-1)),'MaxShift',300);
    tforms{i-1} = T.Tform;
end
save([pathOut filesep 'tforms.mat'],'tforms');
end

%% write

disp('write images, tables')

pfpb = pfpbStart(nImages);
parfor i = 1:nImages
    pfpbUpdate(pfpb);
    imwrite(TS0(:,:,i), [pathOut filesep sprintf('I%03d_C1.tif',i)]);
    imwrite(TS1(:,:,i), [pathOut filesep sprintf('I%03d_C.png',i)]);
    imwrite(TS2(:,:,i), [pathOut filesep sprintf('I%03d_M.png',i)]);
    tiffwriteimj(uint16(65535*TSQ(:,:,i)), [pathOut filesep sprintf('I%03d_CQ.tif',i)]);
    xy = spots5{i};
    writetable(array2table(xy,'VariableNames',{'x','y'}),[pathOut filesep sprintf('I%03d.csv',i)]);
end

%% check

if interactive

%\\\SET
    % index of plane to check
    i = 3;
%///

I = im2double(imread([pathOut filesep sprintf('I%03d_C1.tif',i)]));
Q = im2double(imread([pathOut filesep sprintf('I%03d_CQ.tif',i)]));
C = im2double(imread([pathOut filesep sprintf('I%03d_C.png',i)]));
M = im2double(imread([pathOut filesep sprintf('I%03d_M.png',i)]));
xy = table2array(readtable([pathOut filesep sprintf('I%03d.csv',i)]));
imshow(imadjust(Q)), hold on
plot(xy(:,1),xy(:,2),'o'), hold off
stack = zeros(size(I,1),size(I,2),4);
stack(:,:,1) = imadjust(I); stack(:,:,2) = imadjust(Q); stack(:,:,3) = C; stack(:,:,4) = M;
tlvt(stack)
end

%% write volume for registration to Allen Institute atlas

disp('write volume')
p = pathOut;
l = listfiles(p,'C.png');

for i = 1:length(l)
    disp(i)
    I = im2double(imread(l{i}));
    
%\\\SET
    % resizing factor (dataset pixel size divided by 25)
    % dataset: 0.645 um/pixel
    % allen atlas: 25 um/pixel
    % empirical scale: 10 (compensates for previous downsizing in rsPreProcessing.m)
    I = imresize(I,10*0.645/25);
    % affects section 'read points / masks, transform (similarity)' in rsPlaneAssignment.m
    % affects section 'read, resize template' in rsPlaneAssignment.m
%///
    
    if i == 1
        V = zeros(size(I,1),size(I,2),length(l));
    end
    V(:,:,i) = normalize(imgaussfilt(I,2));
end

volumeWrite(uint8(255*V),[p '.tif']);
disp('done')
if interactive
tlvt(V)
end