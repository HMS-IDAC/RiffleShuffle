function [angles, midPoints, segLengths, strenghts, IOut] = symmetryViaRegistration2D(Image,varargin)
% [angles, midPoints, segLengths, strenghts, IOut] = symmetryViaRegistration2D(Image,varargin)
% Computes mirror symmetry segments in 2D images via registration.
%
% Inputs
%     Image: grayscale or rgb image
% Optional Inputs
%     'RegMethod': the registration method; options are...
%         'dense' (default): uses Matlab's 'imregtform' function to register images
%         'sparse': uses Matlab's 'detectSURFFeatures' and 'matchFeatures' to register images
%         'nxc': uses RANSAC + Normalized Cross Correlation to register images, as described in reference below
%             Note: 'dense' and 'sparse' are fast methods; 'nxc' works best, but takes longer to run.
%     (the following optional parameters are only valid when 'nxc' is used)
%     'BoxSize': dimensions of patch for 'Normalized Cross Correlation registration (default 50)
%     'NumBoxSamples': number of patch samples for RANSAC (default 100)
%     'MaxNumOutputs': suggested maximum number of output symmetry lines; the actuall number can be smaller.
%     'AngleSet': range of rotation angles used by registration algorithm (default 0:6:360-6);
%                 if symmetry axis is known to be nearly vertical, using range around zero (e.g. -30:6:30) might improve results
%
% Outputs
%     angles: angles of symmetry lines
%     midPoints: mid-points of symmetry lines (one per column)
%     segLengths: lenghts of symmetry lines
%     strenghts: strenghts of symmetry lines (they are sorted in descending order, so the first line is always the strongest guess)
%     IOut: image with drawn symmetry lines
%
% Comments
%     1. When 'RegMethod' is 'dense' or 'sparse', only one output line is computed.
%        The only way to get more output lines is by using 'nxc'.
%     2. The angles are in the row/col coordinate system, that is, where x corresponds to rows and y to columns.
%        Therefore, to obtain the end-points p,q of the first symmetry line, do the following:
%            ag = angles(1);
%            mp = midPoints(:,1);
%            sl = segLengths(1);
%            p = mp+sl/2*[cos(ag); sin(ag)]; % one end-point
%            q = mp-sl/2*[cos(ag); sin(ag)]; % another end-point
%        Now the ***row*** of the endpoint p is p(1) and the ***column*** is p(2) 
%        This means that you have invert these coordinates for certain Matlab functions such as insertShape. See example below.
%
% Example
%     Image = imread('gantrycrane.png');
%     Image = imrotate(Image,45,'crop');
% 
%     [angles, midPoints, segLengths, strenghts] = symmetryViaRegistration2D(Image);
% 
%     ag = angles(1);
%     mp = midPoints(:,1);
%     sl = segLengths(1);
%     p = mp+sl/2*[cos(ag); sin(ag)];
%     q = mp-sl/2*[cos(ag); sin(ag)];
%     Image = insertShape(Image,'line',[p(2) p(1) q(2) q(1)],'LineWidth',3,'Color','green');
% 
%     figure
%     image(Image), axis equal, axis off
%
% Reference
%     Finding Mirror Symmetry via Registration
%     Marcelo Cicconet, David G. C. Hildebrand, Hunter Elliott
%     https://arxiv.org/abs/1611.05971
%
% Marcelo Cicconet, Jul 2017

ip = inputParser;
ip.addParameter('RegMethod','dense'); % 'dense','sparse','nxc'
ip.addParameter('BoxSize',50); % only used with 'nxc' registration
ip.addParameter('NumBoxSamples',100); % only used with 'nxc' registration
ip.addParameter('MaxNumOutputs',1); % only used with 'nxc' registration
ip.addParameter('AngleSet',0:6:360-6); % only used with 'nxc' registration
ip.parse(varargin{:});
prmts = ip.Results;

if nargout > 4
    IOut = Image;
end

I = Image;
if size(I,3) == 3
    I = rgb2gray(I);
elseif size(I,3) == 2 || size(I,3) > 3
    I = I(:,:,1);
end
ms = max(size(I));
newImSize = 200;
rf = newImSize/ms;
if rf < 1
    I = imresize(I,rf);
else
    rf = 1;
end

GI = normalize(imgradient(I));
I = normalize(double(I));

if strcmp(prmts.RegMethod,'nxc')
    [cellp,cellv,srmags] = eigsymNXC(GI,0,prmts.BoxSize,prmts.NumBoxSamples,prmts.MaxNumOutputs,prmts.AngleSet);
    srmags = srmags/max(srmags);
    strenghts = srmags(1:length(cellp));
elseif strcmp(prmts.RegMethod,'dense') || strcmp(prmts.RegMethod,'sparse')
    [cellp,cellv] = eigsym(I,0,prmts.RegMethod);
    strenghts = 1;
else
    error('Registration method not recognized.')
end

nOutputs = length(cellp);

angles = zeros(1,nOutputs);
midPoints = zeros(2,nOutputs);
segLengths = zeros(1,nOutputs);

for iOutput = 1:nOutputs
    p = cellp{iOutput};
    v = cellv{iOutput};

    ag = atan2(v(1),v(2));
    if ag < 0
        ag = ag+pi;
    end

    xy = [];
    for j = 1:round(sqrt(2*newImSize^2))
        x = round(p(2)+j*cos(ag));
        y = round(p(1)+j*sin(ag));
        if x >= 1 && x <= size(I,1) && y >= 1 && y <= size(I,2)
            xy = [xy; [x y]];
        end
        x = round(p(2)-j*cos(ag));
        y = round(p(1)-j*sin(ag));
        if x >= 1 && x <= size(I,1) && y >= 1 && y <= size(I,2)
            xy = [xy; [x y]];
        end
    end
    if isempty(xy)
        xy = round(size(I)/2);
    else
        xy = round(mean(xy));
    end
    
    [midpointI,seglenI,~] = endpoints(I,1,ag,xy,'run');
    midpointI = midpointI/rf;
    seglenI = seglenI/rf;
    
    angles(iOutput) = ag;
    midPoints(:,iOutput) = midpointI';
    segLengths(iOutput) = seglenI;
    
    if nargout > 4
        v1 = midpointI+seglenI/2*[cos(ag) sin(ag)];
        v2 = midpointI-seglenI/2*[cos(ag) sin(ag)];
        IOut = insertShape(IOut,'line',[v1(2) v1(1) v2(2) v2(1)],'LineWidth',5,'Color','green');
        IOut = insertShape(IOut,'line',[v1(2) v1(1) v2(2) v2(1)],'LineWidth',1,'Color','yellow');
    end
end

end

% ----------------------------------------------------------------------------------------------------

function [p,v,srmags] = eigsymNXC(I,refAngle,boxSize,nBoxSamples,maxNOutputs,angleSet)
% I should be double, in range [0,1]

% reflect
p = [size(I,2)/2; size(I,1)/2];
N = [cos(refAngle); sin(refAngle)];
d = dot(p,N);
S = [eye(2)-2*(N*N') 2*d*N; 0 0 1];

tform = affine2d(S');
xWorldLimits = [1 size(I,2)];
yWorldLimits = [1 size(I,1)];
J = imwarp(I,tform,'OutputView',imref2d(size(I),xWorldLimits,yWorldLimits));

% register
[tforms, srmags] = computeNormxcorrTransforms(J,I,boxSize,nBoxSamples,maxNOutputs,angleSet);

nTForms = length(tforms);
p = cell(1,nTForms);
v = cell(1,nTForms);
for itform = 1:nTForms
    tform = tforms{itform};

    % compute sym line
    R = tform.T';
    t = R(1:2,3);

    T = S(1:2,1:2)*(R(1:2,1:2)');
    [V,D] = eig(T);
    ieig = [];
    for i = 1:2
        if abs(D(i,i)+1) < 0.000001
            ieig = i;
            break;
        end
    end
    if isempty(ieig)
        error('no -1 eigenvalue')
    end

    v{itform} = V(:,ieig); % eigenvector of eigenvalue -1
    v{itform} = [-v{itform}(2) v{itform}(1)]; % perp

    % point in line
    p{itform} = ((R(1:2,1:2)*(2*d*N))'+t')/2;
end

end

function [tforms, srmags] = computeNormxcorrTransforms(J,I,boxSize,nBoxSamples,maxNOutputs,angleSet) % J: moving, I: fixed

[numrows,numcols] = size(I);
rangles = angleSet;%0:6:360-6;
rmags = zeros(1,length(rangles));
vs = zeros(length(rangles),2);

parfor iangle = 1:length(rangles)
%     if mod(iangle,round(length(rangles)/10)) == 1
%         fprintf('.')
%     end
    A = zeros(2*numrows,2*numcols);
    flipI = imrotate(J,rangles(iangle),'crop');
    for index = 1:nBoxSamples
        w = boxSize;
        h = w;
        x0 = floor((size(flipI,2)-w)*rand);
        y0 = floor((size(flipI,1)-h)*rand);
        xcFlipI = x0+w/2;
        ycFlipI = y0+h/2;

        subFlipI = imcrop(flipI,[x0 y0 w h]);
        if var(subFlipI(:)) > 0.001
            [ROI,~,mc] = locateSubset(subFlipI,I);
            if mc > 0.25
                xcI = ROI(1)+ROI(3)/2;
                ycI = ROI(2)+ROI(4)/2;

                v = [xcI ycI]-[xcFlipI ycFlipI]; % translation

                row = max(min(round(v(2)+numrows),2*numrows),1);
                col = max(min(round(v(1)+numcols),2*numcols),1);
                A(row,col) = A(row,col)+1;
            end
        end
    end
    A = imfilter(A,fspecial('gaussian',12,3));
    maxA = max(A(:));
    
    [r,c] = find(A == maxA);
    v(1) = c(1)-numcols;
    v(2) = r(1)-numrows;
    rmags(iangle) = maxA;
    vs(iangle,:) = v;
end
% fprintf('\n')

[srmags,iangles] = sort(rmags,'descend');
nTForms = min(length(rangles),maxNOutputs);
tforms = cell(1,nTForms);
for i = 1:nTForms
    iangle = iangles(i);
    
    v = vs(iangle,:);

    arad = -rangles(iangle)/360*2*pi;

    % rotation with respect to center
    T1 = [eye(2) [-numcols/2; -numrows/2]; 0 0 1];
    T2 = [[cos(arad) -sin(arad); sin(arad) cos(arad)] [0; 0]; 0 0 1];
    T3 = [eye(2) [numcols/2; numrows/2]; 0 0 1];

    % translation
    T4 = [eye(2) v'; 0 0 1];

    % transform
    tform = affine2d((T4*T3*T2*T1)');
    
    tforms{i} = tform;
end

end

% ----------------------------------------------------------------------------------------------------

function [ROI,c,mc] = locateSubset(subI,I)

c = normxcorr2(subI,I);

mc = max(c(:));
[ypeak, xpeak] = find(c==mc);

yoffSet = ypeak(1)-size(subI,1);
xoffSet = xpeak(1)-size(subI,2);

ROI = [xoffSet+1, yoffSet+1, size(subI,2), size(subI,1)];

end

% ----------------------------------------------------------------------------------------------------

function [cellp,cellv] = eigsym(I,refAngle,regType)
% I should be double, in range [0,1]

% reflect
p = [size(I,2)/2; size(I,1)/2];
N = [cos(refAngle); sin(refAngle)];
d = dot(p,N);
S = [eye(2)-2*(N*N') 2*d*N; 0 0 1];

tform = affine2d(S');
xWorldLimits = [1 size(I,2)];
yWorldLimits = [1 size(I,1)];
J = imwarp(I,tform,'OutputView',imref2d(size(I),xWorldLimits,yWorldLimits));

if strcmp(regType,'dense')
%     optimizer = registration.optimizer.RegularStepGradientDescent;
%     optimizer.MaximumIterations = 500;
%     metric = registration.metric.MeanSquares;
    
    [optimizer,metric] = imregconfig('monomodal');
    tform = imregtform(J,I,'rigid',optimizer,metric);
elseif strcmp(regType,'sparse')
    metricThreshold = 10;
    numOctaves = 4;
    numScaleLevels = 4;

    pointsI = detectSURFFeatures(I,'MetricThreshold',metricThreshold,'NumOctaves',numOctaves,'NumScaleLevels',numScaleLevels);
    [featuresI, pointsI] = extractFeatures(I, pointsI, 'SURFSize', 128);

    pointsJ = detectSURFFeatures(J,'MetricThreshold',metricThreshold,'NumOctaves',numOctaves,'NumScaleLevels',numScaleLevels);
    [featuresJ, pointsJ] = extractFeatures(J, pointsJ, 'SURFSize', 128);

    indexPairs = matchFeatures(featuresJ, featuresI, 'Unique', true);%, 'MaxRatio', 0.1);

    matchedPointsI = pointsI(indexPairs(:,2), :);
    matchedPointsJ = pointsJ(indexPairs(:,1), :);

    if matchedPointsI.Count < 2 % not enough to apply estimateGeometricTransform
        tform = affine2d(eye(3,3));
    else
        tform = estimateGeometricTransform(matchedPointsJ, matchedPointsI,...
            'similarity', 'Confidence', 99.9, 'MaxNumTrials', 2000, 'MaxDistance', 10);
    end
    R = tform.T(1:2,1:2);
    R = 1/sqrt(det(R))*R; % get rid of 'scale' component
    tform.T(1:2,1:2) = R;
end

% compute sym line
R = tform.T';
t = R(1:2,3);

T = S(1:2,1:2)*(R(1:2,1:2)');
[V,D] = eig(T);
ieig = [];
for i = 1:2
    if abs(D(i,i)+1) < 0.000001
        ieig = i;
        break;
    end
end
if isempty(ieig)
    error('no -1 eigenvalue')
end

v = V(:,ieig); % eigenvector of eigenvalue -1
v = [-v(2) v(1)]; % perp

% point in line
p = ((R(1:2,1:2)*(2*d*N))'+t')/2;

cellp = {p};
cellv = {v};

end

% ----------------------------------------------------------------------------------------------------

function [midpointI,seglenI,proximity] = endpoints(I,sigma,angleI,midpointI,mode)

[nr,nc] = size(I);

imcent = round([nr nc]/2);
d = imcent-midpointI;
I = imtranslate(I,[d(2) d(1)]);
I = imrotate(I,-180*angleI/pi,'crop');

freq = 1/sigma;
index = 0;
anglerange = [-pi/3 -pi/6 0 pi/6 pi/3];
Convs = complex(zeros(nr,nc,length(anglerange)));
for langle = pi/2+anglerange
    index = index+1;
    [mr,mi] = cmorlet(sigma,freq,langle,0);
    J = conv2(I,mr+1i*mi,'same');
    Convs(:,:,index) = J;
end

hnc = floor(nc/2);
SS = zeros(nr,hnc);
for i = 1:index
   L = Convs(:,1:hnc,i);
   R = Convs(:,end-hnc+1:end,length(anglerange)+1-i);

   R = fliplr(R);
   S = abs(L.*conj(R));

   SS = max(SS,S);
end

proximity = sum(sum(SS)); % proximity between half images

SortS = normalize(sort(SS,2,'descend'));
SortS = SortS(:,1:10);

s = sum(SortS,2);

l = 5;
k1 = [ones(1,l) -ones(1,l)];
s1 = conv(s,k1,'same');
if strcmp(mode,'debug')
    subplot(1,2,1)
    plot(s,'linewidth',2), hold on
    plot(s1,'linewidth',2)
end
[pks,lcs] = findpeaks(max(s1,0),'NPeaks',1,'MinPeakHeight',0.05*max(abs(s1)));
i0 = lcs;
if strcmp(mode,'debug')
    plot(i0,pks,'ok','linewidth',2)
end
[pks,lcs] = findpeaks(max(-flipud(s1),0),'NPeaks',1,'MinPeakHeight',0.05*max(abs(s1)));
i1 = length(s)-lcs+1;
if strcmp(mode,'debug')
    plot(i1,-pks,'ok','linewidth',2), hold off, axis off
end
if strcmp(mode,'debug')
    subplot(1,2,2)
    imshow([normalize(SS) SortS I])
    pause
end

ep0 = [i0 round(nc/2)];
ep1 = [i1 round(nc/2)];

R = [cos(angleI) -sin(angleI); sin(angleI) cos(angleI)];
ep0 = round(R*(ep0-imcent)'+imcent'-d');
ep1 = round(R*(ep1-imcent)'+imcent'-d');

midpointI = round(0.5*(ep0+ep1))';
seglenI = norm(ep1-ep0);

end

% ----------------------------------------------------------------------------------------------------

function [mr,mi] = cmorlet(sigma,freq,angle,halfkernel)

% ref: http://arxiv.org/pdf/1203.1513.pdf, page 2

support = 2.5*sigma;

xmin = -support;
xmax = -xmin;
ymin = xmin;
ymax = xmax;
xdomain = xmin:xmax;
ydomain = ymin:ymax;

[x,y] = meshgrid(xdomain,ydomain);

xi = freq*[sin(angle); cos(angle)];

envelope = exp(-0.5*(x.*x+y.*y)/sigma^2); 
carrier = exp(1i*(xi(1)*x+xi(2)*y));

% makes sum of args = 0
C2 = sum(sum(envelope.*carrier))/sum(sum(envelope));

% makes sum of args*conj(args) = 1
arg = (carrier-C2).*envelope;
normfact = sum(sum(arg.*conj(arg)));
C1 = sqrt(1/normfact);

psi = C1*(carrier-C2).*envelope;

if halfkernel
    condition = ((xi(1)*x+xi(2)*y) <= 0);
    mr = real(psi).*condition;
    mi = imag(psi).*condition;
else
    mr = real(psi);
    mi = imag(psi);
end

end

% ----------------------------------------------------------------------------------------------------

function J = normalize(I)
    J = I-min(min(I));
    J = J/max(max(J));
end