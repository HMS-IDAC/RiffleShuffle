function F = fadeLR(I,f)

c0 = round(f*size(I,2));
c1 = round((1-f)*size(I,2));
F = I;
for c = 1:c0
    f = (c-1)/c0;
    F(:,c) = f*F(:,c);
end
for c = c1:size(F,2)
    f = 1-(c-c1)/(size(F,2)-c1);
    F(:,c) = f*F(:,c);
end

end