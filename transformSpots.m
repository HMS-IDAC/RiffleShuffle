function txy = transformSpots(xy,tform)
    
% size(xy) = n points rows, 2 columns; first column x (column location), second column y (row location)
T = tform.T';
xy3 = [xy'; ones(1,size(xy,1))];
xy3 = T*xy3;
txy = xy3(1:2,:)';

end