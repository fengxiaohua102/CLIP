function meas_data = SPC_4DLF_F_ACC(image_object, code, opt)
    % This is the accelerated version: recovering 4D light field version
    % implement the forward operator for CLIP-SPC
    % image_object: [m,n,Na,Na]
    % code:[n,n,Na,Na,N_meas]
    % meas_data: [Na,Na,N_meas]

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
        image_object = reshape(image_object,[opt.n_size,4*opt.n_size]);      
    else            
        image_object = reshape(image_object,[m,n,Na,Na]);
    end 
    
    image_object = transf_opt( image_object );  
    
    if(GPU_ACC)
        meas_data = single(zeros(Na,Na,N_meas,'gpuArray'));
    else
        meas_data = zeros(Na,Na,N_meas);
    end

    for K_x = 1:Na
        for K_y = 1:Na
            for K_m = 1:N_meas                
                % 2: apply SPC operator
                im_coded = squeeze(image_object(:,:,K_x,K_y)) .* squeeze(code(:,:,K_x,K_y,K_m));
                meas_data(K_x,K_y,K_m) = sum(im_coded(:)); 
            end
        end
    end
end