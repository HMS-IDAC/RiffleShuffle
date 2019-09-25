%% setup

clear, clc

%\\\SET
    % full path to folder containing images
    pathIn = '/scratch/RiffleShuffle/Stacks/KMR16_Sorted_B_1.55_2.45';

    % path to contour and mask machine learning models
    pathModelC = '/scratch/RiffleShuffle/SupportFiles/modelC.mat';
    pathModelM = '/scratch/RiffleShuffle/SupportFiles/modelM.mat';

    % if to quantify spots (otherwise quantifies diffuse signal)
    quantSpots = true; 
%///

l1 = listfiles(pathIn,'_C1.tif');
l2 = listfiles(pathIn,'_C2.tif');
load(pathModelC);
load(pathModelM);

%% measure spot

%\\\SET
    % index of image to use to test spot parameters
    i = 1;
%///
    
I2 = imread(l2{i});
J2 = imresize(I2,0.5);
spotMeasureTool(normalize(im2double(J2)));

%% test spot detection parameters (no need to run if not quantifying spots)

%\\\SET
    % index of image to use to test spot detection
    i = 1;
    
    % sigma of spot (measured above)
    sigma = 3.6;
    
    % 'distance to background distribution' threshold; decrease to detect more spots (range [0,~100])
    dist2BackDistThr = 5;
    
    % 'similarity to ideal spot' threshold; decrease to select more spots (range [-1,1])
    spotinessThreshold = 0.8;
%///

I = imread(l1{i});
I2 = imread(l2{i});

J = imresize(I,0.1);
J2 = imresize(I2,0.5);

doubleJ = normalize(double(J));
L = pixelClassifierClassify(doubleJ,modelMask);
Mask = bwareafilt(L == 2,[0.01*numel(L) Inf]);

[~,ptSrcImg] = logPSD(J2, imresize(Mask,size(J2),'nearest'), sigma, dist2BackDistThr);
ptSrcImg = selLogPSD(J2, ptSrcImg, sigma, spotinessThreshold);

[r,c] = find(ptSrcImg);

J2 = imresize(J2,0.2);
r = r/5;
c = c/5;

imshow(imadjust(J2)), hold on
plot(c,r,'o'), hold off


%% downsizes images, detects spots, writes to file

pathOut = [pathIn '_Downsized'];
if ~exist(pathOut,'dir')
    mkdir(pathOut);
end

pfpb = pfpbStart(length(l1));
parfor i = 1:length(l1)
    I = imread(l1{i});
    I2 = imread(l2{i});
    
    J = imresize(I,0.1);
    J2 = imresize(I2,0.5);

    doubleJ = normalize(double(J));
    [~,contourPM] = pixelClassifierClassify(doubleJ,modelC);
    contourPM = contourPM(:,:,1);
    
    L = pixelClassifierClassify(doubleJ,modelM);
    Mask = bwareafilt(L == 2,[0.01*numel(L) Inf]);

    if quantSpots
    [~,ptSrcImg] = logPSD(J2, imresize(Mask,size(J2),'nearest'), sigma, dist2BackDistThr);
    ptSrcImg = selLogPSD(J2, ptSrcImg, sigma, spotinessThreshold);
    [r,c] = find(ptSrcImg);
    r = r/5;
    c = c/5;
    end
  
    % _C1.tif: channel used to find contours
    % _CQ.tif: channel to quantify (instead of quantifying spots)
    % _M.png: Mask
    % _C.png: Contour probability maps
    % .csv: spot locations
    imwrite(uint8(255*doubleJ),[pathOut filesep sprintf('I%03d_C1.tif',i)]);
    tiffwriteimj(imresize(J2,size(J)),[pathOut filesep sprintf('I%03d_CQ.tif',i)]);
    imwrite(255*uint8(Mask),[pathOut filesep sprintf('I%03d_M.png',i)]);
    imwrite(uint8(255*contourPM),[pathOut filesep sprintf('I%03d_C.png',i)]);
    if quantSpots
    writetable(array2table([r c],'VariableNames',{'r','c'}),[pathOut filesep sprintf('I%03d.csv',i)]);
    end
    
    pfpbUpdate(pfpb);
end

%% check saved files

figureQSS
for i = 1:length(l1)
    I = imread([pathOut filesep sprintf('I%03d_C1.tif',i)]);
    Q = imread([pathOut filesep sprintf('I%03d_CQ.tif',i)]);
    M = imread([pathOut filesep sprintf('I%03d_M.png',i)]);
    C = imread([pathOut filesep sprintf('I%03d_C.png',i)]);
    A = table2array(readtable([pathOut filesep sprintf('I%03d.csv',i)]));
    
    subplot(1,4,1)
    imshow(imadjust(I)), title('I')
    
    subplot(1,4,2)
    imshow(imadjust(Q))
    if quantSpots
        hold on, plot(A(:,2),A(:,1),'o'), hold off
    end
    title('Q')
    
    subplot(1,4,3)
    imshow(M), title('M')
    
    subplot(1,4,4)
    imshow(C), title('C')
    pause%(0.1)
end
close all