clear, clc

p = '/home/mc457/files/CellBiology/IDAC/Marcelo/Sabatini/Hochbaum/Registration/sampleAnnotatedBrain/DataForPC/Mask';
modelM = pixelClassifierTrain(p);

%% save model

save('/home/mc457/files/CellBiology/IDAC/Marcelo/Etc/RiffleShuffle/SupportFiles/modelM.mat','modelM');

%%

imPaths = listfiles(p,'.tif');

i = 13;
I = imreadGrayscaleDouble(imPaths{i});
L = pixelClassifierClassify(I,modelM);
Mask = bwareafilt(L == 2,[0.01*numel(L) Inf]);
switchBetween(I,Mask)