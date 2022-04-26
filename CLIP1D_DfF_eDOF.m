%************
% This script simulate the CLIP depth retrieval 
% using the dpeth from defocus method
% Method: LIFT
%************
clear all; clc; close all;
% read raw orignial image
data_loader = load([pwd '\scene5_LF_Data_LIFT.mat']);
angles = data_loader.angles; N_dec = 1;
angles = angles(:,:,1:N_dec:end);
meas_data = data_loader.meas_data;  %[n,Na,Na,N_meas]
meas_data = meas_data(:,:,:,1:N_dec:end);
LF_data = data_loader.LF_sim;  % ground truth 4D light field

Nx = data_loader.Nx;
cnt = data_loader.cnt;

[n,~,Na,N_meas] = size(meas_data);
% 3: process the measurement
meas_data = meas_data./max(meas_data(:));
% meas_data = meas_data(:);

%% SPC solver setup
% 1: setup model 
opt_model.basis = 'dwavelet';
opt_model.wavelet_op = 'sym4';  % wavelet type: db, sym, haar coif etc.
opt_model.n = n;
opt_model.Nrefx = Na/2;
opt_model.Nrefy = Na/2;
if(strcmp(opt_model.basis, 'wavelet'))
    [a_flt,~,~,~] = wfilters(opt_model.wavelet_op);
    wavelet_len = length(a_flt)/2-1;
    n_size = ceil(n/2)+wavelet_len;
    opt_model.n_size = n_size;   
end

% 2: setup solver
opt.maxiter = 100;       % param for max iteration
opt.lambda = 10e0;      % param for regularizing param
opt.vis = -1;
opt.denoiser = 'mixed';   % option of denoiser: BM3D, ProxTV
opt.POScond = false;       % positiveness contraint on the solution
global GLOBAL_useGPU;
GLOBAL_useGPU = 1;
opt.monotone = 1;

%% DfF step1: generate focal stack
Ndepth = 24;
s_array = linspace(-1.0,0.6,Ndepth);
im_focal_stack = zeros(Nx,Nx,Ndepth);
for K_s = 1: length(s_array)   
    opt_model.s = s_array(K_s);
    disp('calculating step size...');
    A = @(x) LIFT_F_Transf(x, angles, opt_model);
    AT = @(x) LIFT_T_Transf(x, angles, opt_model);

    if(strcmp(opt_model.basis, 'wavelet'))
        max_egival = power_iter(A,AT,zeros(n_size,n_size*4))
    else
        max_egival = power_iter(A,AT,zeros(n,n))
    end
 
    opt.step = 1.0*max_egival;  % step size

    % 3: call solver
    switch (opt_model.basis)
        case 'dct'
            disp('use dct transform')
            x0 = zeros(n,n);
            [x, convergence] = Solver_PlugPlay_FISTA2D(A, AT, meas_data, x0, opt);
            x = reshape(x, [n,n]);
            im_recon = idct2(x);
        case 'wavelet'        
            x0 = zeros(n_size,4*n_size);
            [x, convergence] = Solver_PlugPlay_FISTA2D(A, AT, meas_data, x0, opt);
            x = reshape(x, [n_size,4*n_size]);
            im_recon = my_inv_wavelet_tranform(x, opt_model.wavelet_op);
        otherwise
            if(K_s ==1)
                x0 = zeros(n,n);
            else
                x0 = im_recon;
            end
            [x, convergence] = Solver_PlugPlay_FISTA2D(A, AT, meas_data, x0, opt);
            im_recon = reshape(x, [n,n]);
    end
    im_focal_stack(:,:,K_s) = norm1(im_recon( cnt-Nx/2:cnt+Nx/2-1, cnt-Nx/2:cnt+Nx/2-1));
end
% im_focal_stack = norm1(im_focal_stack);

%% DfF step 2: focal stack filtering
[~,im_focal_stack_flt] = VBM3D(im_focal_stack,5);
figure;
subplot(1,2,1); montage(im_focal_stack); colormap('hot'); axis equal; axis off;title('Org');
subplot(1,2,2); montage(im_focal_stack_flt); colormap('hot'); axis equal; axis off;title('Filtered');

%% DfF Step 3: calculate relative depth
[depth_fit, depth_map, Img_ALLFOCUS] = depth_from_focus( ...
                                       im_focal_stack_flt, 'SML', 0.1, 0, 0);

figure(2); ax(1)=subplot(1,2,1); imagesc(depth_map,[0,max(depth_map(:))]); colormap(ax(1), 'jet'); axis equal; axis off;
ax(2)=subplot(1,2,2); imagesc(Img_ALLFOCUS); colormap(ax(2),'hot'); axis equal; axis off;


%%
s_array = [-1.0, -0.4, 0.1];
% opt.lambda = 0.01e-1;      % param for regularizing param
% opt.denoiser = 'BM3D';   % option of denoiser: BM3D, ProxTV
for K_s =1:length(s_array)
    opt_model.s = s_array(K_s);
    A = @(x) LIFT_F_Transf(x, angles, opt_model);
    AT = @(x) LIFT_T_Transf(x, angles, opt_model);
    max_egival = power_iter(A,AT,zeros(n,n)) 
    opt.step = 1.0*max_egival;  % step size
    x0 = zeros(n,n);
    [x, convergence] = Solver_PlugPlay_FISTA2D(A, AT, norm1(meas_data), x0, opt);
    im_recon = reshape(x, [n,n]);
    im_refocs(:,:,K_s) = norm1(im_recon( cnt-Nx/2:cnt+Nx/2-1, cnt-Nx/2:cnt+Nx/2-1));
end
%
figure('Renderer', 'painters', 'Position', [600 600 500 200]);
[ha,pos] = tight_subplot(1,3,[.01 .01],[.05 .05],[.01 .01]);
for K_s =1:length(s_array)
    axes(ha(K_s)); imagesc(im_refocs(2:end-1,2:end-1,K_s));
%     title(['s: ' num2str(s_array(K_s))]);
    colormap('hot'); axis off; axis square;
end
im_focal_stack_t = norm1(LFrefocus( LF_data, s_array));
% calculate NMSE
for K = 1:size(im_focal_stack_t,3)
    [n_mse, ~] = eval_quality(im_focal_stack_t(4:end-4,4:end-4,K), im_refocs(4:end-4,4:end-4,K))
end

% % 3D visulization of the reconstructed scene
% [x,y]= meshgrid(1:128,1:128);
% % figure(4); colormap('hot');pcshow([x(:),y(:),depth_map_TV(:)*100],Img_ALLFOCUS(:),'MarkerSize',100);
% figure(4); surf(x,y,depth_map_TV*100,Img_ALLFOCUS, 'LineStyle','None'); colormap('hot');
% ax= gca; set(ax,'Color','k'); ax.XAxis.Visible = 'off';
% ax.YAxis.Visible = 'off'; ax.ZAxis.Visible = 'off';
% view([0,1,1]);

% point_cloud_ab = [[x(:),y(:),depth_map_TV(:)*100],Img_ALLFOCUS(:)];
% T = array2table(point_cloud_ab);
% T.Properties.VariableNames(1:4) = {'x_axis','y_axis','z_axis','Mag'};
% writetable(T, ['LF_scene1.csv']);


