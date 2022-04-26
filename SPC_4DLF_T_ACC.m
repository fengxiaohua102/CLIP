function img_recon = SPC_4DLF_T_ACC(meas_data, code, opt)
    % implement the adjoint operator of CLIP-SPC
    % meas_data:[Na,Na,N_meas]
    % img_recon: [m,n,Na,Na]
    % code:[n,n,Na,Na,N_meas]

    GPU_ACC = isa(meas_data, 'gpuArray');
    switch (opt.basis)
        case 'dct'
            transf_opt = @(x) dct2(x);
        case 'wavelet'        
            wavelet_op = opt.wavelet_op;
            transf_opt = @(x) my_wavelet_tranform(x,wavelet_op);
        otherwise
            transf_opt = @(x) x;
    end
    
    [m,n,~,Na,N_meas] = size(code);   % number of measurment   
    if(GPU_ACC)
        img_recon = single(zeros(m,n,Na,Na,'gpuArray'));
        im_sum = single(zeros(m,n,'gpuArray'));
    else
        img_recon = zeros(m,n,Na,Na);
        im_sum = zeros(m,n);
    end
    
    for K_x = 1:Na
        for K_y = 1:Na
            im_sum = im_sum .*0;
            for K_m = 1:N_meas                
                % 1: apply SPC operator
                im_coded = meas_data(K_x,K_y,K_m) .* squeeze(code(:,:,K_x,K_y,K_m));
                im_sum = im_sum + im_coded; 
            end
            img_recon(:,:,K_x,K_y) = im_sum;
        end
    end
    
    % 2: apply transform basis operator
    img_recon = transf_opt(img_recon);
end