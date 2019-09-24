function graystackmontage(S)
    M = zeros(size(S,1),size(S,2),1,size(S,3));
    for i = 1:size(S,3)
        M(:,:,1,i) = S(:,:,i);
    end
    montage(M)
end