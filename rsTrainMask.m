clear, clc

p = '/scratch/RiffleShuffle/DataForPC/Mask';
modelM = pixelClassifierTrain(p);

%% save model

save('/scratch/RiffleShuffle/SupportFiles/modelM.mat','modelM');

%%

imPaths = listfiles(p,'.tif');

i = 1;
I = imreadGrayscaleDouble(imPaths{i});
L = pixelClassifierClassify(I,modelM);
Mask = bwareafilt(L == 2,[0.01*numel(L) Inf]);
switchBetween(I,Mask)