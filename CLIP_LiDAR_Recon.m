%% *************
% **** This script perform LiDAR-based NLOS Geometry Calibration *******
% ****** It calculates wall 3D and laser spot 3D (x, y, z) matrix********

% Wall 3D location--Time zero calib
close all; clear all; clc;
top_DIR = [pwd '\LiDAR\' ]; 

calib_DIR = [top_DIR '\CalibData\'];
DIR = [top_DIR ];
filename = 'scene5'; %18 17

USE_HIS = false;  % flag to indicate the datafile is a HIS
lambda_TV = 3e-1; % the hyperparameter for the FISTA reconstruction
IMG_INV = false;  % use image relay where image is inverted

% ******
% 1: Load the calibration data results
% ******
Calib_Res = load([calib_DIR,'Calib_Res.mat']);
Calib_Res = Calib_Res.Calib_Res;
streak_tform = Calib_Res.streak_tform_10ns_real;
% system-secific parameters
r_resolution = 20e-12*3e8/2 * 1e3;   % the radial depth resolution (ct/2), in mm;
time_zero_offset = 539.2163 %%% 610-------- Initialization to 800
%time_zero_offset = 537.8258; %%% 0607 0608
scale_ratio = 10; % dx/cdt

% ******
% 2: Load image data and extract the slice useful for image reconstruction
% ******
% backGround = double(imread([calib_DIR 'backgroundCalib_MCP20.tif']));
if(USE_HIS)
    reconFile_HIS = [DIR '\' filename '.his'];  %
    [image_uncorrected, image_t] = readHIStoOne(reconFile_HIS,'SUM', true, 200);
%      H = readHIS(reconFile_HIS);  % images = H.getAllFrame();   image_t = sum(images,3);
else
    %image_t = double(imread([DIR filename '\' filename '.tif']));
    image_t = double(imread([DIR '\' filename '.tif']));
end
backG = mean(mean(image_t(1000,200:1000,:)));
image_t = image_t-backG;
image_t(image_t<=0) = 0 ;
image_t = imwarp(image_t, streak_tform, 'OutputView', imref2d(size(image_t)));

image_t(1:20,:) = 0;  image_t(1000:end,:) = 0;

% make the distant signal stronger
end_t = 900; start_t = 800;
image_t = image_t(150:end_t,:,:);
image_t = diag( (start_t:(start_t-1+size(image_t,1))).^2 ) * double(image_t);
figure; imagesc(mean(image_t,3)); colormap('hot'); title('Corrected image');

%% 3: setup solver
u = (1:7)-4; s = 0; % shearing factor for refocusing
% sub_img_cnt = round(Calib_Res.sub_img_cnt + u.*s); % normal refoucsing procedure
% disabling the dpeth-fof-field version: supplementary note 3.3
sub_img_cnt = round(Calib_Res.sub_img_cnt + u.*(s.*tan(deg2rad(90-Calib_Res.Angle))).');
options = []; options.INVERT = IMG_INV; options.CROP = true;
options.Deconv = false; options.USE_TV = false; options.Refocus = true;
options.sub_img_cnt = round(sub_img_cnt); 
options.Normalize = true;

im_crop = fx_LIFT_ReconVOL(Calib_Res, image_t, lambda_TV, options);
im_crop = gather(im_crop);
% filtering the noises
% [~,im_crop] = VBM3D(im_crop,10);
% im_crop(isnan(im_crop)) = 0.0;
Nx_LIFT = size(im_crop,1);

% Extract the depth-index and MPI (maximum amplitude projection)
[MPI_im, idx_im] = E_MAG_t(im_crop);
% [MPI_im, idx_im] = max(im_crop,[],3);

figure; subplot(1,2,1); imagesc(MPI_im); colormap('hot'); axis square;
subplot(1,2,2); imagesc(idx_im); colormap('hot'); axis square;

% generate a point cloud in matlab without producing absolute coordinates
Nx_IM_half = round((size(im_crop,1)-1)/2);
[x ,y] = meshgrid(-Nx_IM_half:Nx_IM_half, -Nx_IM_half:Nx_IM_half);

point_xyz = zeros( Nx_LIFT, Nx_LIFT, 3);
point_xyz(:,:,1) = x .* scale_ratio;  point_xyz(:,:,2) = y .* scale_ratio; 
point_xyz(:,:,3) = (idx_im + time_zero_offset) .* r_resolution;
point_cloud = [reshape(point_xyz,[],3), reshape(MPI_im,[],1)];

% ptCld = pointCloud(point_xyz, 'Intensity', sqrt(MPI_im));
% ptCld = pcdenoise(ptCld,'NumNeighbors',6);
% figure; ax = pcshow(ptCld); colormap('hot');
T = array2table(point_cloud);
T.Properties.VariableNames(1:4) = {'x_axis','y_axis','z_axis','Mag'};
writetable(T, [DIR '\' filename '.csv']);
%% Caltulate & write the point cloud's absolute coord to a cvs file for paraview
params = load([calib_DIR,'CameraMatrix.mat']);
params = params.params;
cam_K = params.Intrinsics.IntrinsicMatrix;
cam_K = cam_K.';

[u ,v] = meshgrid(1:size(im_crop,1), 1:size(im_crop,1));
point_uv = zeros( size(im_crop,1), size(im_crop,2), 3);
point_uv(:,:,1) = u;  point_uv(:,:,2) = v;  point_uv(:,:,3) = 1; 
point_z = reshape( point_xyz(:,:,3), [],1);
point_coo = reshape(point_uv,[],3);
point_coo = point_coo.'; % the shape being [3, N_points]
% Calculate absolute values
point_coo = inv(cam_K) * point_coo;

% Convert the radial distance of LiDAR to a rectangular coo.
point_coo_r = vecnorm(point_coo,2,1).';
point_z = point_z ./ point_coo_r;

% reshape to [N_point, 3]
point_coo = point_coo * diag(point_z); % [3, N_points]

point_int = reshape(MPI_im, [], 1);
point_cloud_ab = [point_coo.', (norm1(point_int))];
min_z = min(point_cloud_ab(:,3)); point_cloud_ab(1,3) = min_z-200;
point_cloud_ab(end,3) = max(point_cloud_ab(:,3))+200;
% point_cloud_ab(point_cloud_ab(:,3)<min_z + 00,4) = NaN; 
point_cloud_ab(:,3) = point_cloud_ab(:,3)*0.5;
ptCld = pointCloud(reshape(point_coo.',Nx_LIFT,Nx_LIFT,3), 'Intensity', sqrt(MPI_im));
ptCld = pcdenoise(ptCld,'NumNeighbors',6);
figure; ax = pcshow(ptCld); colormap('hot');

T = array2table(point_cloud_ab);
T.Properties.VariableNames(1:4) = {'x_axis','y_axis','z_axis','Mag'};
writetable(T, [DIR '\' filename '.csv']);

%% generate a focal stack for the LiDAR scene
% u = (1:7)-4; s_array = -3:0.25:3; lambda_TV = 3e-1;
% MPI_im_refocus = zeros(Nx_LIFT,Nx_LIFT,length(s_array));
% s_array = s_array([5,10,15,20]);
% for  L =1:length(s_array) % shearing factor for refocusing
%     % disabling the dpeth-fof-field version: supplementary note 3.3
%     s = s_array(L);
%     sub_img_cnt = round(Calib_Res.sub_img_cnt + u.*(s.*tan(deg2rad(90-Calib_Res.Angle))).');
%     options.Refocus = true;
%     options.sub_img_cnt = round(sub_img_cnt); 
% 
%     im_crop = fx_LIFT_ReconVOL(Calib_Res, image_t, lambda_TV, options);
%     im_crop = gather(im_crop);
% 
%     % Extract the depth-index and MPI (maximum amplitude projection)
%     [MPI_im, idx_im] = E_MAG_t(im_crop);
% %     MPI_im_refocus(:,:,L) = MPI_im;
%     figure; imagesc(MPI_im); colormap('hot'); axis square;
% 
% end
% % filtering the noises
% % [~,MPI_im_refocus_flt] = VBM3D(MPI_im_refocus,10);
% % MPI_im_refocus_flt = MPI_im_refocus;
% % MPI_im_refocus_flt(isnan(MPI_im_refocus_flt)) = 0.0;
% % figure; montage(MPI_im_refocus_flt); colormap('hot');
