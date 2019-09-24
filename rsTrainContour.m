clear, clc

p = '/home/mc457/files/CellBiology/IDAC/Marcelo/Sabatini/Hochbaum/Registration/sampleAnnotatedBrain/DataForPC/Contour';
modelC = pixelClassifierTrain(p,'adjustContrast',true,'adjustContrastThr',0.01);

%% save model

save('/home/mc457/files/CellBiology/IDAC/Marcelo/Etc/RiffleShuffle/SupportFiles/modelC.mat','modelC');

%%

imPaths = listfiles(p,'.tif');

i = 5;
I = imreadGrayscaleDouble(imPaths{i});
[~,P] = pixelClassifierClassify(I,modelC);
imshow([imadjust(I), P(:,:,1)])