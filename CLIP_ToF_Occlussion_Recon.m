%% *************
% **** This script perform CLIP-TOF imaging through occlusion ********
% **** reproducing the video imaging results *************************
close all; clear all; clc;
top_DIR = [pwd '\Occlusion\']

calib_DIR = [top_DIR '\CalibData\'];
DIR = [top_DIR];
filename = 'og2'; %18 17
lambda_TV = 2e-1; % the hyperparameter for the FISTA reconstruction
IMG_INV = false;  % use image relay where image is inverted

% ******
% 1: Load the calibration data results
% ******
Calib_Res = load([calib_DIR,'Calib_Res.mat']);
Calib_Res = Calib_Res.Calib_Res;
streak_tform = Calib_Res.streak_tform_2ns;

% system-secific parameters
r_resolution = 2e-12*3e8/2 * 1e3;   % the radial depth resolution (ct/2), in mm;
time_zero_offset = 60/r_resolution-312; %%% 610-------- Initialization to 800
scale_ratio = 20/60; % dx/cdt
image_t_loader = load([DIR '\' filename '.mat']);
image_t = image_t_loader.image_t;

%% select region of interest in the time domain to reduce running time
image_t2 = image_t(240:640,:,:);
figure; imagesc(image_t2(:,:,200));
%% 3: setup solver
lambda_TV = 3e-1; % the hyperparameter for the FISTA reconstruction
u = (1:7)-4; 
s = 0; % shearing factor for refocusing
% disabling the depth-fof-field version: supplementary note 3.3
% sub_img_cnt = round(Calib_Res.sub_img_cnt + u.*(s.*tan(deg2rad(90-Calib_Res.Angle))).');
options = []; options.INVERT = IMG_INV; options.CROP = true;
options.Deconv = false; options.USE_TV = false; options.Refocus = true;
options.Normalize = false; 
options.sub_img_cnt = round(Calib_Res.sub_img_cnt + u.*s); % normal refoucsing procedure

tic
const_val = double(max(image_t2(:)));
s_idx = 66; N_avg = 8;
for K = 1:floor(size(image_t2,3)*0.5)-N_avg
%     image_tt = double(image_t(:,:,K)); %image_ds(:,:,K);
    image_tt = 0;
    for K_avg = 1:N_avg
        image_tt = image_tt + 1/N_avg .* double( image_t2(:,:,2*(K-1)+K_avg) ); 
    end
    image_tt = image_tt./const_val;
    image_ttt = image_tt(1:3:end-2,:) + image_tt(2:3:end-1,:) + image_tt(3:3:end,:);
    s_range = zeros(size(image_ttt,1),1);
    s_range(1:s_idx) = 5.5; s_range(s_idx+1:end) = -6; % shearing
    options.s_range = s_range;
    
    tic
    [im_crop,im_deconv] = fx_LIFT_ReconVOL_DOF(Calib_Res, image_ttt, lambda_TV, options);
    toc
    im_crop = gather(im_crop); im_deconv = gather(im_deconv);
    
%     im_video(:,:,:,K) = norm1(single(im_crop));
%     im_crop(:,:,1:s_idx) = im_crop(:,:,1:s_idx)*0.5;
    norm_const = max(reshape(im_crop(:,:,1:s_idx),[],1));% 
    im_crop = (im_crop-min(im_crop(:)))./norm_const;
    norm_const = max(reshape(im_deconv(:,:,1:s_idx),[],1));% 
    im_deconv = (im_deconv-min(im_deconv(:)))./norm_const;


    vtkwrite([DIR filename '\'  filename '_LiDAR_Recon' num2str(K) '.vtk'], ...
        'structured_points', 'LiDAR_Vol', im_crop);
    vtkwrite([DIR filename '\'  filename '_LiDAR_Recon_deconv' num2str(K) '.vtk'], ...
        'structured_points', 'LiDAR_Vol', im_deconv);

end
toc
% Extract the depth-index and MPI (maximum amplitude projection)
[MPI_im, idx_im] = E_MAG_t(im_crop); [MPI_im2, idx_im2] = E_MAG_t(im_deconv);
figure; subplot(2,2,1); imagesc(MPI_im); colormap('hot'); axis square;
subplot(2,2,2); imagesc(idx_im); colormap('hot'); axis square;
subplot(2,2,3); imagesc(MPI_im2); colormap('hot'); axis square;
subplot(2,2,4); imagesc(idx_im2); colormap('hot'); axis square;

