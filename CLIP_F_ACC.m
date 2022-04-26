function meas_data = CLIP_F_ACC(image_object_1d, code, opt)
    % This is the accelerated version
    % implement the forward operator for CLIP-SPC
    % image_object: [m,n]
    % code:[n,n,Na,Na,N_meas]
    % meas_data: [Na,Na,N_meas]
    % s: the shearing factor
    s = -opt.s;
    Nrefx = opt.Nrefx;
    Nrefy = opt.Nrefy;
    GPU_ACC = isa(code, 'gpuArray');
    [m,n,~,Na,N_meas] = size(code);  
    switch (opt.basis)
        case 'dct'
            transf_opt = @(x) idct2(x);
        case 'wavelet'        
            wavelet_op = opt.wavelet_op;
            transf_opt = @(x) my_inv_wavelet_tranform(x,wavelet_op);
        otherwise
            transf_opt = @(x) x;
    end
    if( strcmp(opt.basis, 'wavelet'))
        image_object = reshape(image_object_1d,[opt.n_size,4*opt.n_size]);      
    else            
        image_object = reshape(image_object_1d,[m,n]);
    end 
    
    image_object = transf_opt( image_object );  
    
    if(GPU_ACC)
        meas_data = single(zeros(Na,Na,N_meas,'gpuArray'));
    else
        meas_data = zeros(Na,Na,N_meas);
    end

    [vv, uu] = ndgrid(1:m, 1:n);
    for K_x = 1:Na
        nx = (s*(K_x-Nrefx));
        for K_y = 1:Na
            % 1: apply shearing operator             
            ny = (s*(K_y-Nrefy));
%             image_sheared = my_shift(image_object,nx,ny);  
            % 1b: interpolation
            image_sheared = interp2(image_object, uu+ny, vv+nx, 'cubic',mean(image_object(:)));
            for K_m = 1:N_meas                
                % 2: apply SPC operator
                im_coded = image_sheared .* squeeze(code(:,:,K_x,K_y,K_m));
                meas_data(K_x,K_y,K_m) = sum(im_coded(:)); 
            end
        end
    end
    meas_data = meas_data(:);
end