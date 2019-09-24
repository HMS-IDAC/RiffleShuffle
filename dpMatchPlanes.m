function optimalAssignmentIndices = dpMatchPlanes(imSubSeq,imSeq,scaleRange,vertDispRange,initialMatch,maxShift,displayProgress,displayOutput)

nSS = length(imSubSeq);
nS = length(imSeq);
CR = zeros(nSS,nS);
lCR = cell(1,nSS*nS);
ij = cell(nSS*nS,2);
count = 0;
for i = 1:nSS
    for j = 1:nS
        count = count+1;
        ij{count,1} = i;
        ij{count,2} = j;
    end
end

if displayProgress
    pfpb = pfpbStart(count);
end
parfor idx = 1:count
    if displayProgress
        pfpbUpdate(pfpb);
    end
    i = ij{idx,1};
    j = ij{idx,2};
    imSS = imSubSeq{i};
    imS = imSeq{j};
    
    if abs(initialMatch(i)-j) > maxShift
        c2 = 0;
    else
        [~,c2] = imscalereg(imSS,imS,scaleRange,scaleRange,vertDispRange);
    end
%     [TI,c2] = imscalereg(imSS,imS,rg,rg);
%     imshowlt(imadjust(TI),imadjust(imS))
%     pause, close all

    lCR{idx} = c2;
end

count = 0;
for i = 1:nSS
    for j = 1:nS
        count = count+1;
        CR(i,j) = lCR{count};
    end
end

C = 1-CR;

[nRows, nCols] = size(C);

if nRows < 2
    error('not enough rows')
end
if nCols < 2
    error('not enough cols')
end

pathCosts = zeros(size(C));

pathCosts(1,:) = C(1,:);
predecessors = nan(size(C));

for i = 2:nRows
    for j = 1:nCols
        localCosts = pathCosts(i-1,:);
        localCosts(j:end) = Inf;
        [m,im] = min(localCosts);
        updatedCost = m+C(i,j);
        pathCosts(i,j) = updatedCost;
        predecessors(i,j) = im;
    end
end

[~,im] = min(pathCosts(nRows,:));
cPath = zeros(nRows,1);
cPath(nRows) = im;
for i = nRows-1:-1:1
    cPath(i) = predecessors(i+1,cPath(i+1));
end

optimalAssignmentIndices = cPath;

if displayOutput
    P = zeros(size(C));
    for i = 1:nRows
        P(i,cPath(i)) = 1;
    end
    subplot(2,1,1)
    imshow(imresize(CR,10,'nearest')), title('correlations, mapped to [0, 1]')
    subplot(2,1,2)
    imshow(imresize(P,10,'nearest')), title('optimal assignment')
end

end