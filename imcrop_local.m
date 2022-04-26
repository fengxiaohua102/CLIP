function image = imcrop_local(image, cnt, half_size)
% crop the image beginning in the center cnt and half image size of half_size
image = image(cnt-half_size:cnt+half_size,cnt-half_size:cnt+half_size,:);
end