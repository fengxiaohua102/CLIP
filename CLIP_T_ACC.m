function img_recon = CLIP_T_ACC(meas_data, code, opt)
    % implement the adjoint operator of CLIP-SPC
    % meas_data:[Na,Na,N_meas]
    % img_recon: [m,n]
    % code:[n,n,Na,Na,N_meas]
    % s: the shearing factor
    s = -opt.s;
    Nrefx = opt.Nrefx;
    Nrefy = opt.Nrefy;
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
    meas_data = reshape(meas_data, [Na,Na,N_meas]);
    if(GPU_ACC)
        img_recon = single(zeros(m,n,'gpuArray'));
        im_sum = single(zeros(m,n,'gpuArray'));
    else
        img_recon = zeros(m,n);
        im_sum = zeros(m,n);
    end
    
    [vv, uu] = ndgrid(1:m, 1:n);
    for K_x = 1:Na
        nx = (-s*(K_x-Nrefx));
        for K_y = 1:Na
            im_sum = im_sum .*0;
            ny = (-s*(K_y-Nrefy));
            for K_m = 1:N_meas                
                % 1: apply SPC operator
                im_coded = meas_data(K_x,K_y,K_m) .* squeeze(code(:,:,K_x,K_y,K_m));
                im_sum = im_sum + im_coded; 
            end
            % 2: apply shearing operator
%             img_recon = img_recon + my_shift(im_sum,nx, ny);
            img_recon = img_recon + interp2(im_sum, uu+ny, vv+nx,'cubic', 0 ); % mean(im_sum(:))
        end
    end
    
    % 2: apply transform basis operator
    img_recon = transf_opt(img_recon);
%     img_recon = img_recon(:);
end