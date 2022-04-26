% ************* This script perform LIFT reconstruciton for NLOS *******************
% ****** It performs the reconstruction in a volumetric (all frames) manner ********
% ****** and the (x,y,t) datacube is then unwarped ***********
close all; clear all; clc;
% top_DIR = 'D:\Dropbox\Xiaohua_Feng\Lidar_NLOS\0406\'  
% DIR = [top_DIR '\CalibData4\nlos10\'];
% calib_DIR = [top_DIR '\CalibData4\'];
calib_DIR = [pwd '\NLOS\'];
filename = 'Y4';
USE_HIS = true;  % flag to indicate the datafile is a HIS
lambda_TV = 0.5e-1;   % the hyperparameter for the FISTA reconstruction
IMG_INV = false;  % use image relay where image is inverted
zeroline = 100;
% ******
% 1: Load the calibration data results
% ******
Calib_Res = load([calib_DIR,'Calib_Res.mat']);
Calib_Res = Calib_Res.Calib_Res;
streak_tform = Calib_Res.streak_tform_10ns_real;

% ******
% 2: Load image data and extract the slice useful for image reconstruction
% ******
if(USE_HIS)
    reconFile_HIS = [calib_DIR filename '.his'];  %
    %bkg_HIS = [DIR  '0610\bkg1.his'];  %
    % Read HIS file with internal jitter correction enabled
     [image_t_uncrorrect, image_t] = readHIStoOne(reconFile_HIS,'SUM', true, 200); 
    % [bkgimage_t_uncrorrect, bkgimage_t] = readHIStoOne(bkg_HIS,'SUM', true, 200); 
%      H = readHIS(reconFile_HIS); image_t = H.getFrame(4);
else
    image_t = double(imread([DIR filename '.tif']));
end

image_t = imwarp(image_t, streak_tform, 'OutputView', imref2d(size(image_t)));
backG = mean(mean(image_t(900:1000,200:1000)));
image_t = image_t-backG;
image_t(image_t<0) = 0 ;

figure; subplot(1,2,1); imagesc(norm1(image_t_uncrorrect)); colormap('jet');
title('Uncorrected streak image');
subplot(1,2,2); imagesc(norm1(image_t)); colormap('jet');
title('Jitter-corrected and Rectified streak image');    
image_t(1:zeroline,:)=0; 
image_t = image_t(1:350,:);
figure; imagesc(norm1(image_t));

%% 3: setup solver
u = (1:7)-4; s = 0; % shearing factor for refocusing
% sub_img_cnt = round(Calib_Res.sub_img_cnt + u.*s); % normal refoucsing procedure
% disabling the dpeth-fof-field version: supplementary note 3.3
sub_img_cnt = round(Calib_Res.sub_img_cnt + u.*(s.*tan(deg2rad(90-Calib_Res.Angle))).');
options = []; options.INVERT = IMG_INV; options.CROP = true;
options.Deconv = false; options.USE_TV = false;
options.Refocus = true; options.sub_img_cnt = round(sub_img_cnt); 
options.Normalize = true;

im_crop = fx_LIFT_ReconVOL(Calib_Res, image_t, lambda_TV, options);
im_crop = gather(im_crop);
% tic
% [im_crop, image_meas]= LIFT_iradonRecon3D(Calib_Res, image_t, options);
% toc
figure; subplot(1,2,1); imagesc(sum(im_crop,3)); title('Wall DC image');
% use inverse radon transform to reconstruct the time-of-flight data
% *********
%% 4: unwarp the NLOS data
% % *********
dt = 10e-12;   DeltaT = 3e8*dt;
% temporal resolution in the time window: 4.6 ps or 10 ps 
[nlos_data] = nlos_unwarp(im_crop, Calib_Res, dt);
% nlos_data(:,1:70,:) = 0;
subplot(1,2,2); imagesc(sum(nlos_data,3)); title('Croed Wall DC image');
camera_grid_pos = Calib_Res.grid_pos; laser_grid_pos = Calib_Res.laser_pos;
% save the results for NLOS reconstrcution
save([calib_DIR filename '_iradon.mat'], 'nlos_data', ...
    'DeltaT','camera_grid_pos','laser_grid_pos');
