function [J,r0,c0] = centerCrop(I,w,h)

r0 = floor(size(I,1)/2-h/2);
c0 = floor(size(I,2)/2-w/2);
J = I(r0+1:r0+h,c0+1:c0+w);

end