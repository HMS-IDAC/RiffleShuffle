function [P,offset] = padTB(I,f)
   
P = zeros(round(f*size(I,1)),size(I,2));
offset = floor((size(P,1)-size(I,1))/2);
P(offset+1:offset+size(I,1),:) = I;
offset = offset+1;

end