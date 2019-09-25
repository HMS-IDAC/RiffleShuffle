function [imageList,labelList,classIndices,adjustContrastThrOut] = pcParseLabelFolder(dirPath,balanceClasses,adjustContrast,adjustContrastThrIn)
% reads images and generates class-balanced labels from annotations

files = dir(dirPath);
adjustContrastThrOut = adjustContrastThrIn;

% how many annotated images (all that don't have 'Class' in the name)
nImages = 0;
for i = 1:length(files)
    fName = files(i).name;
    if ~files(i).isdir && ...
       ~contains(fName,'Class') && ...
       ~contains(fName,'.mat') && ...
       ~contains(fName,'.db') && ...
       fName(1) ~= '.'
        nImages = nImages+1;
        imagePaths{nImages} = [dirPath filesep fName];
    end
end

% list of class indices per image
classIndices = [];
[~,imName] = fileparts(imagePaths{1});
for i = 1:length(files)
    fName = files(i).name;
    k = strfind(fName,'Class');
    if contains(fName,imName) && ~isempty(k)
        [~,imn] = fileparts(fName);
        classIndices = [classIndices str2double(imn(k(1)+5:end))];
    end
end
nClasses = length(classIndices);

% read images/labels
imageList = cell(1,nImages);
labelList = cell(1,nImages);
if isempty(adjustContrastThrOut)
    pct99 = zeros(1,nImages);
end
for i = 1:nImages
    I = imreadGrayscaleDouble(imagePaths{i});
    if adjustContrast
        pct99(i) = prctile(I(:),99);
        I = imadjust(I);
    end
    [imp,imn] = fileparts(imagePaths{i});
    
    if balanceClasses
        nSamplesPerClass = zeros(1,nClasses);
    end
    lbMaps = cell(1,nClasses);
    for j = 1:nClasses
        classJ = imread([imp filesep imn sprintf('_Class%d.png',classIndices(j))]);
        classJ = (classJ(:,:,1) > 0);
        if balanceClasses
            nSamplesPerClass(j) = sum(classJ(:));
        end
        lbMaps{j} = classJ;
    end
    
    if balanceClasses
        [minNSamp,indMinNSamp] = min(nSamplesPerClass);
    end
    
    L = uint8(zeros(size(I)));

    for j = 1:nClasses
        if balanceClasses
            if j ~= indMinNSamp
                classJ = lbMaps{j} & (rand(size(classJ)) < minNSamp/nSamplesPerClass(j));
            else
                classJ = lbMaps{j} > 0;
            end
        else
            classJ = lbMaps{j} > 0;
        end
        L(classJ) = j;
    end


    imageList{i} = I;
    labelList{i} = L;
end
if isempty(adjustContrastThrOut)
    adjustContrastThrOut = 0.1*mean(pct99);
end

end