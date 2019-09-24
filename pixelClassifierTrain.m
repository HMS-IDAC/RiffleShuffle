function model = pixelClassifierTrain(trainPath,varargin)
% model = pixelClassifierTrain(trainPath,varargin)
% trains a single-layer random forest for image segmentation (pixel classification)
% use the 'pixelClassifier' function to apply the model
% see ...IDAC_git/common/imageSegmentation/PixelClassifier/ReadMe.txt for more details
%
% trainPath
% where images and labes are;
% images are assumed to have the same size;
% every image should have the same number of accompanied label masks,
% labeled <image name>_ClassX.png, where X is the index of the label;
% labels can be created using ImageAnnotationBot:
% https://www.mathworks.com/matlabcentral/fileexchange/64719-imageannotationbot
%
% ----- varargin ----- 
%
% sigmas, default [1 2 4 8]
% basic image features are simply derivatives (up to second order) in different scales;
% this parameter specifies such scales; details in pcImageFeatures.m
%
% offsets, default []
% in pixels; for offset features (see pcImageFeatures.m)
% set to [] to ignore offset features
% 
% osSigma, default 2
% sigma for offset features
%
% radii, default []
% range of radii on which to compute circularity features (see pcImageFeatures.m)
% set to [] to ignore circularity features
%
% cfSigma, default 2
% sigma for circularity features
%
% logSigmas, default []
% sigmas for LoG features (see pcImageFeatures.m)
% set to [] to ignore LoG features
%
% sfSigmas, default []
% steerable filter features sigmas (see pcImageFeatures.m)
% set to [] to ignore steerable filter features
% 
% nhoodEntropy, default []
% local entropy filter neighborhood sizes (see pcImageFeatures.m)
% must be an array of odd numbers (to satisfy entropyfilt.m restrictions)
% set to [] to ignore local entropy filter features
% 
% nhoodStd, default []
% local standard deviation filter neighborhood sizes (see pcImageFeatures.m)
% must be an array of odd numbers (to satisfy stdfilt.m srestrictions)
% set to [] to ignore local standard deviation features
%
% nTrees, default 20
% number of decision trees in the random forest ensemble
%
% minLeafSize, default 60
% minimum number of observations per tree leaf
%
% pctMaxNPixelsPerLabel, default 1
% percentage of max number of pixels per label (w.r.t. num of pixels in image);
% this puts a cap on the number of training samples and can improve training speed
%
% balanceClasses, default true
% if to balance number of pixels per label
%
% ----- output ----- 
% model
% structure containing model parameters
%
% ...
%
% Marcelo Cicconet, Dec 11 2017 (release)
%
% Clarence Yapp, Apr 3 2018 (added local entropy and std features)
%
% Marcelo Cicconet, Aug 17 2018 (added contrast adjustment option)

%% parameters

ip = inputParser;
ip.addParameter('sigmas',[1 2 4 8]);
ip.addParameter('offsets',[]);
ip.addParameter('osSigma',2);
ip.addParameter('radii',[]);
ip.addParameter('cfSigma',2);
ip.addParameter('logSigmas',[]);
ip.addParameter('sfSigmas',[]);
ip.addParameter('nhoodEntropy',[]);
ip.addParameter('nhoodStd',[]);
ip.addParameter('edgeSigmas',[]);
ip.addParameter('edgenangs',16);
ip.addParameter('ridgeSigmas',[]);
ip.addParameter('ridgenangs',16);
ip.addParameter('nTrees',20);
ip.addParameter('minLeafSize',60);
ip.addParameter('pctMaxNPixelsPerLabel',1);
ip.addParameter('balanceClasses',true);
ip.addParameter('adjustContrast',true);
ip.addParameter('adjustContrastThr',[]);
% if 'adjustContrast' is true and 'adjustContrastThr' is empty,
% 'adjustContrastThr is computed automatically as 0.1 of the average 99th percentiles of training images;
% at test time, if 'adjustContrast' is true images are only adjusted if prctile(image,99) > 'adjustContrastThr'

ip.parse(varargin{:});
p = ip.Results;

sigmas = p.sigmas;
offsets = p.offsets;
osSigma = p.osSigma;
radii = p.radii;
cfSigma = p.cfSigma;
logSigmas = p.logSigmas;
sfSigmas = p.sfSigmas;
nhoodEntropy = p.nhoodEntropy;
nhoodStd = p.nhoodStd;
edgeSigmas = p.edgeSigmas;
edgenangs = p.edgenangs;
ridgeSigmas = p.ridgeSigmas;
ridgenangs = p.ridgenangs;
nTrees = p.nTrees;
minLeafSize = p.minLeafSize;
pctMaxNPixelsPerLabel = p.pctMaxNPixelsPerLabel;
balanceClasses = p.balanceClasses;
adjustContrast = p.adjustContrast;
adjustContrastThr = p.adjustContrastThr;

%% read images/labels

[imageList,labelList,labels,adjustContrastThr] = pcParseLabelFolder(trainPath,balanceClasses,adjustContrast,adjustContrastThr);
nLabels = length(labels);

%% training samples cap

maxNPixelsPerLabel = (pctMaxNPixelsPerLabel/100)*size(imageList{1},1)*size(imageList{1},2);
nImages = length(imageList);
for imIndex = 1:nImages
    L = labelList{imIndex};
    for labelIndex = 1:nLabels
        LLI = L == labelIndex;
        nPixels = sum(sum(LLI));
        rI = rand(size(L)) < maxNPixelsPerLabel/nPixels;
        L(LLI) = 0;
        LLI2 = rI & (LLI > 0);
        L(LLI2) = labelIndex;
    end
    labelList{imIndex} = L;
end

%% construct train matrix

ft = [];
lb = [];
tic
for imIndex = 1:nImages
    fprintf('computing features from image %d of %d\n', imIndex, nImages);
%     [F,featNames] = pcImageFeatures(imageList{imIndex},sigmas,offsets,osSigma,radii,cfSigma,logSigmas,sfSigmas,[],[],[],[],nhoodEntropy,nhoodStd);
    [F,featNames] = pcImageFeatures(imageList{imIndex},sigmas,offsets,osSigma,radii,cfSigma,logSigmas,sfSigmas,ridgeSigmas,ridgenangs,edgeSigmas,edgenangs,nhoodEntropy,nhoodStd);
    L = labelList{imIndex};
    [rfFeat,rfLbl] = rfFeatAndLab(F,L);
    ft = [ft; rfFeat];
    lb = [lb; rfLbl];
end
fprintf('time spent computing features: %f s\n', toc);

%% training

fprintf('training...'); tic
[treeBag,featImp,oobPredError] = rfTrain(ft,lb,nTrees,minLeafSize);
figureQSS
subplot(1,2,1), barh(featImp), set(gca,'yticklabel',featNames'), set(gca,'YTick',1:length(featNames)), title('feature importance')
subplot(1,2,2), plot(oobPredError), title('out-of-bag classification error')
fprintf('training time: %f s\n', toc);

%% model

model.treeBag = treeBag;
model.sigmas = sigmas;
model.offsets = offsets;
model.osSigma = osSigma;
model.radii = radii;
model.cfSigma = cfSigma;
model.logSigmas = logSigmas;
model.sfSigmas = sfSigmas;
model.nhoodEntropy = nhoodEntropy;
model.nhoodStd = nhoodStd;
model.ridgeSigmas = ridgeSigmas;
model.ridgenangs = ridgenangs;
model.edgeSigmas = edgeSigmas;
model.edgenangs = edgenangs;
model.adjustContrast = adjustContrast;
model.adjustContrastThr = adjustContrastThr;

disp('done training')

end