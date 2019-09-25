clear, clc

p = '/scratch/RiffleShuffle/DataForPC/Contour';
modelC = pixelClassifierTrain(p,'adjustContrast',true,'adjustContrastThr',0.01);

%% save model

save('/scratch/RiffleShuffle/SupportFiles/modelC.mat','modelC');

%%

imPaths = listfiles(p,'.tif');

i = 5;
I = imreadGrayscaleDouble(imPaths{i});
[~,P] = pixelClassifierClassify(I,modelC);
imshow([imadjust(I), P(:,:,1)])