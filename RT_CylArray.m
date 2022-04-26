function img_sum = RT_CylArray(image_F, angle, n,Nt)
    % implement the adjoint operator of tomographic photography
    % angle is the array specifying the angle in degrees of integration in each lenslet
    % line_image:(N_angle,n);
    % image_F: a vector of [n*N_angle*Nt(Nt),1]
    % image_t: [n,n,Nt]
    
    GPU_ACC = isa(image_F, 'gpuArray');
    
    N_angle = length(angle);
    line_image = image_F;
    % generate mask to account for the rotation operation that may
    % generate shapes larger than original image
    x = ((1:n)-n/2)/n;     y = x;
    [x,y] = meshgrid(x,y);
    mask = (x.^2+y.^2)>=0.25;
    mask = repmat(mask,[1,1,Nt]);
    
    if(GPU_ACC)
        mask = gpuArray(mask);
        img_sum = single(zeros(n,n,Nt,'gpuArray'));
    else
        img_sum = zeros(n,n,Nt);
    end
    
    line = permute(line_image,[2,1,3]); % (N_angle,n,Nt)

    for K = 1:N_angle
        image_y = repmat(line(K,:,:),[n,1,1]);  %[n,n,Nt]       
        img_sum = img_sum + imrotate(image_y, -angle(K),'bilinear', 'crop');
    end
    img_sum(mask) = 0.;    
end