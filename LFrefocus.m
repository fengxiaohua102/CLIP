function im_refocus = LFrefocus(LF_in,s_array)
% Own implementation for light field refocusing
[Nax,Nay,m,n] = size(LF_in);
[vv, uu] = ndgrid(1:m, 1:n);
Nrefx = floor((Nax+1)/2); Nrefy = floor((Nay+1)/2);
im_refocus = zeros(m,n,length(s_array));

for L = 1:length(s_array)
    s = s_array(L);
    for K_x = 1:Nax
        nx = (s*(K_x-Nrefx));
        for K_y = 1:Nay
            ny = (s*(K_y-Nrefy));
            im_view = squeeze(LF_in(K_x,K_y,:,:)); %medfilt2( squeeze(LF_in(K_x,K_y,:,:)), [5,5]);
            image_sheared = interp2( im_view, uu+ny, vv+nx, 'cubic',median(im_view(:)) );
            im_refocus(:,:,L) = im_refocus(:,:,L) + image_sheared;
        end
    end
end
