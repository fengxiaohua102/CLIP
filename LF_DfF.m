%% This script performs depth from focus with conventional unfocused 
% light field cameras

clear all; clc; close all;
DIR = 'D:\CLIP\';
data_loader = load([DIR 'scene4_LF_Data_LIFT.mat']);
LF_data = data_loader.LF_sim;
Nx = size(LF_data, 3);
%% try to focus on different depths 

Ndepth = 24;
s_array = linspace(-1.0,0.4,Ndepth);
im_focal_stack = zeros(Nx,Nx,Ndepth);
for K = 1: length(s_array)
    [ImgOut, FiltOptions, LF2] = LFFiltShiftSum( LF_data, s_array(K));
    im_focal_stack(:,:,K) = norm1(ImgOut(:,:,1));
end
% im_focal_stack = (im_focal_stack);

%% DfF step 2: focal stack filtering
[~,im_focal_stack_flt] = VBM3D(im_focal_stack,10);
figure;
subplot(1,2,1); montage(im_focal_stack); colormap('hot'); axis equal; axis off;title('Org');
subplot(1,2,2); montage(im_focal_stack_flt); colormap('hot'); axis equal; axis off;title('Filtered');

%% DfF Step 3: calculate relative depth
[depth_fit, depth_map, Img_ALLFOCUS] = depth_from_focus( ...
                                       im_focal_stack_flt, 'SML', 0.5, 0, 0);
figure; subplot(1,3,1); imagesc( depth_fit); colormap('jet'); axis equal; axis off;
subplot(1,3,2); imagesc( depth_map); colormap('jet'); axis equal; axis off;
ax = subplot(1,3,3); imagesc( Img_ALLFOCUS); colormap(ax,'hot'); axis equal; axis off;


s_array = [-0.60, -0.1, 0.2];
for K_s =1:length(s_array)    
    [ImgOut, FiltOptions, LF2] = LFFiltShiftSum( LF_data, s_array(K_s));
    im_refocus(:,:,K_s) = norm1(ImgOut(:,:,1));
end
%
figure('Renderer', 'painters', 'Position', [600 600 500 200]);
[ha,pos] = tight_subplot(1,3,[.01 .01],[.05 .05],[.01 .01]);
for K_s =1:length(s_array)
    axes(ha(K_s)); imagesc(im_refocus(2:end-1,2:end-1,K_s));
%     title(['s: ' num2str(s_array(K_s))]);
    colormap('hot'); axis off; axis square;
end