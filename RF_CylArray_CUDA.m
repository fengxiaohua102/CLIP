function image_F = RF_CylArray_CUDA(image_cube,angle, k_cu_kernal)
    % This is the accelerated version
    % implement the forward operator that of tomographic photography
    % angle is the two d array specifying the angle in degrees of integration in each lenslet
    % image_cube: [n,n,Nt]
    % image_F: a vector of [n*N_angle*Nt(Nt),1]
    
    N_angle = length(angle);
    [~,n,Nt] = size(image_cube);
    
    line_image_t = single(zeros(n,Nt,N_angle,'gpuArray'));
    im = single(zeros(size(image_cube),'gpuArray'));
    for K = 1:N_angle
        im_r = imrotate_gpu(k_cu_kernal, image_cube, im, angle(K));
        line_image_t(:,:,K) = sum(im_r,1); 
    end
    line_image_t = permute(line_image_t,[1,3,2]);
    % line_image shape: [n,N_angle,Nt]
    image_F = line_image_t;
end