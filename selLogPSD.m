function sPsImage = selLogPSD(I,psImage,sigma,thr)
% selects spots by quality
%
% inputs
% ------
% I: image
% psImage: point source image mask, obtained with logPSD
% sigma: std dev of gaussian that best fits spots
% thr: correlation threshold; spot is selected if correlation with ideal spot is above this; should be in range [-1,1]
%
% output
% ------
% point source image mask (false everywhere except at location of selected spots)

hks = round(2*sigma);
k = zeros(2*hks+1,2*hks+1);
k(hks+1,hks+1) = 1;
k = imgaussfilt(k,sigma);
k = k/sum(k(:));
k = k(:);

sPsImage = false(size(psImage));
DI = im2double(I);
[r,c] = find(psImage);
for i = 1:length(r)
    r0 = r(i)-hks;
    c0 = c(i)-hks;
    r1 = r(i)+hks;
    c1 = c(i)+hks;
    if r0 >= 1 && r1 <= size(I,1) && c0 >= 1 && c1 <= size(I,2)
        P = DI(r0:r1,c0:c1);
        P = P/sum(P(:));
        if corr(k,P(:)) > thr
            sPsImage(r(i),c(i)) = 1;
        end
    end
end

end