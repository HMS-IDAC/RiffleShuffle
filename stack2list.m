function l = stack2list(S)

l = cell(1,size(S,3));
for i = 1:size(S,3)
    l{i} = S(:,:,i);
end
    
end