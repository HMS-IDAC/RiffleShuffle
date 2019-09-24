function [L,P] = pixelClassifierClassify(I,model)
% I should be grayscale double (see imreadGrayscaleDouble.m)

II = I;
if model.adjustContrast && prctile(II(:),99) > model.adjustContrastThr
    II = imadjust(I);
end
if ~isempty(model.ridgeSigmas) && ~isempty(model.edgeSigmas)
    F = pcImageFeatures(II,model.sigmas,model.offsets,model.osSigma,model.radii,model.cfSigma,model.logSigmas,model.sfSigmas,model.ridgeSigmas,model.nRidgeAngs,model.edgeSigmas,model.nEdgeAngs,model.nhoodEntropy,model.nhoodStd);
else
    F = pcImageFeatures(II,model.sigmas,model.offsets,model.osSigma,model.radii,model.cfSigma,model.logSigmas,model.sfSigmas,[],[],[],[],model.nhoodEntropy,model.nhoodStd);
end
[L,P] = imClassify(F,model.treeBag,100);

end