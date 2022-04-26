function image_t = LIFT_4DLF_T(image_F, angle, opt)
    % implement the adjoint operator of LIFT : recovering 3D/4D light field version
    % angle is the array specifying the angle in degrees of integration in each lenslet
    % image_F: a vector of [n,Na,Na,N_meas], from forward operator
    % image_t: [n,n], the 'recon' images/coeffs.
    %****
    % it differs in providing transform (DCT or wavelet) operation
    %****
    % wavelet coeff are concated as [ca, cb, cc, cd]  
    
    n = opt.n;
    [Na,Na,N_meas] = size(angle);   % the LIFT angle arrangements  
    line_image = reshape(image_F,[n,Na,Na,N_meas]);
    switch (opt.basis)
        case 'dct'
            transf_opt = @(x) dct2(x);
        case 'wavelet'        
            wavelet_op = opt.wavelet_op;            
            transf_opt = @(x) my_wavelet_tranform(x,wavelet_op);
        otherwise
            transf_opt = @(x) x;
    end

    % generate mask
    x = ((1:n)-n/2)/n;y = x; [x,y] = meshgrid(x,y); mask = (x.^2+y.^2)>=0.24;
    image_recon = zeros(n,n,Na,Na);
    for K_x = 1: Na
        for K_y = 1: Na
            % 1: apply LIFT operator 
            img_sum = zeros(n,n);
            for K_m = 1:N_meas
                line = line_image(:,K_x,K_y,K_m); % [n,Na,Na,N_meas] 
                image_y = repmat(reshape(line,[1,n]),[n,1]); 
                img_sum = img_sum + imrotate(image_y, -angle(K_x,K_y,K_m),'bilinear', 'crop');
            end       
            image_recon(:,:,K_x,K_y)= img_sum;
        end
    end        
%     image_recon(mask) = 0.;

    % 2: apply transform basis operator
    image_t = transf_opt(image_recon);
%     image_t = image_t(:);
end