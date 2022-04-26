function image_F = LIFT_4DLF_F(image_1d,angle,opt)
    % implement the forward operator of LIFT: recovering 3D/4D light field
    % angle is the two d array specifying the angle in degrees of integration in each lenslet
    % image_1d: [n*n*na,1] --> could be a image of coeff. in a basis
    % For wavelet basis, image_cube: [n, 4*n]
    % image_F: a vector of [n*N_angle,1]
    %****
    % it differs in providing transform (DCT or wavelet) operation
    %****
    n = opt.n;
    switch (opt.basis)
    case 'dct'
        transf_opt = @(x) idct2(x);
    case 'wavelet'        
        wavelet_op = opt.wavelet_op;
        transf_opt = @(x) my_inv_wavelet_tranform(x,wavelet_op);
        n_size = opt.n_size;
    otherwise
        transf_opt = @(x) x;
    end 
    [Na,Na,N_meas] = size(angle);   % the LIFT angle arrangements 
    if( strcmp(opt.basis, 'wavelet'))
        image = reshape(image_1d,[n_size,4*n_size]);      
    else            
        image = reshape(image_1d,[n,n,Na,Na]);  
    end 
    
    line_image = zeros(n,Na,Na,N_meas);
%     x = ((1:n)-n/2)/n; y = x; [x,y] = meshgrid(x,y); mask = (x.^2+y.^2)>=0.24;
    % 1: apply the transformation
    image_space = transf_opt(image); 
%     image_space(mask) = 0;
    % 3: apply LIFT operator
    for K_x = 1: Na
        for K_y = 1: Na
            image_shear = image_space(:,:,K_x,K_y);
            for K_m = 1:N_meas
                im = imrotate(image_shear,angle(K_x,K_y,K_m),'bilinear', 'crop');
                line_image(:,K_x,K_y,K_m) = sum(im,1).'; 
            end
        end
    end
    % line_image shape: [n,Na,Na,N_meas]
    image_F = line_image;
end