function img_sum = RT_CylArray_CUDA(image_F,angle, n, Nt, k_cukernel)
    % implement the adjoint operator of tomographic photography
    % angle is the array specifying the angle in degrees of integration in each lenslet
    % line_image:(N_angle,n);
    % image_F: a vector of [n*N_angle*Nt(Nt),1]
    % image_t: [n,n,Nt]
  
    N_angle = length(angle);
    line_image = image_F;    
    img_sum = single(zeros(n,n,Nt,'gpuArray'));
    img_r = single(zeros(n,n,Nt,'gpuArray'));
    line = permute(line_image,[2,1,3]); % (N_angle,n,Nt)

    for K = 1:N_angle
        image_y = repmat(line(K,:,:),[n,1,1]);  %[n,n,Nt]       
        img_sum = img_sum + imrotate_gpu(k_cukernel, image_y, img_r, -angle(K));
    end
end