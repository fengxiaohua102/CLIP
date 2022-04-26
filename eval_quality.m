function [n_mse, ssim_val] = eval_quality(Img_ref,im_recon)
    Img_ref = Img_ref./norm(Img_ref(:));
    im_recon = im_recon./norm(im_recon(:));
    
    N_t = 400; NMSE = zeros(N_t,1); ssimval = zeros(N_t,1);
    Img_ref_zm = Img_ref - mean(Img_ref(:));
    im_recon_zm = im_recon - mean(im_recon(:));
    for K = 1:N_t
        alpha = 1+(K-N_t/2)*0.003;
        im_recon_n = im_recon_zm .* alpha;
        err = (Img_ref_zm - im_recon_n);
        NMSE(K) = norm(err(:)).^2 ./norm(Img_ref_zm(:)).^2;
        
%         ssimval(K) = ssim(im_recon_n,double(Img_ref_zm) );
    end
    n_mse = min(NMSE);
    ssim_val = 0;%max(ssimval);
end