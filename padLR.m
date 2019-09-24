function [P,offset] = padLR(I,f)
   
P = zeros(size(I,1),round(f*size(I,2)));
offset = floor((size(P,2)-size(I,2))/2);
P(:,offset+1:offset+size(I,2)) = I;
offset = offset+1;

end