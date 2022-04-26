%************
% This script simulate the CLIP depth retrieval 
% using the dpeth from defocus method: SPC implementation
%************
clear all; clc; close all;
%% read raw orignial image
data_loader = load([pwd '\scene1_LF_Data_SPC.mat']);
codes = data_loader.codes; N_dec = 8;
codes = codes(:,:,:,:,1:N_dec:end);
meas_data = data_loader.meas_data;  %[Na,Na,N_meas]
meas_data = meas_data(:,:,1:N_dec:end);
[m,n,~,Na,N_meas] = size(codes);
LF_data = data_loader.LF_sim;

% preprocess the codes and measurement data
codes_temp = reshape(codes, [m*n,Na*Na*N_meas]);
for K  = 1: size(codes_temp,1)
    codes_temp(K,:) = codes_temp(K,:) - mean( codes_temp(K,:) );
end
fac = 1.0/mean(vecnorm(codes_temp,2,2));
codes = reshape(codes_temp.*fac, [m,n,Na,Na,N_meas]);
% 3: process the measurement
meas_data = meas_data - mean(meas_data(:));
meas_data = meas_data./max(meas_data(:));
meas_data = meas_data(:);

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
opt.lambda = 5e-2;      % param for regularizing param
opt.vis = -1;
opt.denoiser = 'ProxTV';   % option of denoiser: BM3D, ProxTV
opt.POScond = false;       % positiveness contraint on the solution
global GLOBAL_useGPU;
GLOBAL_useGPU = 1;
opt.monotone = 1;

%% DfF step1: generate focal stack
Ndepth = 24;
s_array = linspace(-1.0,0.4,Ndepth);
im_focal_stack = zeros(n,n,Ndepth);
for K_s = 1: length(s_array)   
    opt_model.s = s_array(K_s);
    disp('calculating step size ...');
    A = @(x) CLIP_F_ACC(x, codes, opt_model);
    AT = @(x) CLIP_T_ACC(x, codes, opt_model);
%     option 1: RED
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
    im_focal_stack(:,:,K_s) = im_recon;
   
    % option 2: damp
%     im_focal_stack(:,:,K_s) = norm1((DAMP(double(meas_data)*1,30,m,n,'TV',A,AT) ) );
end
% im_focal_stack = norm1(im_focal_stack);

%% DfF step 2: focal stack filtering
im_focal_stack_flt = zeros(size(im_focal_stack));
im_focal_stack_flt(2:end-1, 2:end-1,: ) = im_focal_stack(2:end-1, 2:end-1,: );
[~,im_focal_stack_flt] = VBM3D(im_focal_stack_flt,5);
for K = 1:size(im_focal_stack_flt,3)
    im_focal_stack_flt(:,:,K) = norm1(im_focal_stack_flt(:,:,K));
end

figure;
subplot(1,2,1); montage(im_focal_stack); colormap('hot'); axis equal; axis off;title('Org');
subplot(1,2,2); montage(im_focal_stack_flt); colormap('hot'); axis equal; axis off;title('Filtered');

%% DfF Step 3: calculate relative depth
[depth_fit, depth_map, Img_ALLFOCUS] = depth_from_focus( ...
                                       im_focal_stack_flt, 'SML', 0.1, 0, 0);
figure; ax1=subplot(1,2,1); imagesc( depth_map,[0,max(depth_map(:))]); colormap(ax1,'jet'); axis equal; axis off;
ax2=subplot(1,2,2); imagesc( Img_ALLFOCUS);  axis equal; axis off; colormap(ax2,'hot');

%% generate a few refocused images
s_array2 = [-1.0, -0.4, 0.3];
im_refocs = zeros(n,n,length(s_array2));
for K_s =1:length(s_array2)
    opt_model.s = s_array2(K_s);
    disp('calculating step size...');
    A = @(x) CLIP_F_ACC(x, codes, opt_model);
    AT = @(x) CLIP_T_ACC(x, codes, opt_model);
    max_egival = power_iter(A,AT,zeros(n,n)) ;
    opt.step = 1.0*max_egival;  % step size
    x0 = zeros(n,n);
    [x, convergence] = Solver_PlugPlay_FISTA2D(A, AT, (meas_data), x0, opt);
    im_recon = reshape(x, [n,n]);
    im_refocs(:,:,K_s) = im_recon;
end
im_focal_stack_t = norm1(LFrefocus( LF_data, s_array2));
% calculate NMSE
for K = 1:size(im_focal_stack_t,3)
    [n_mse, ~] = eval_quality(im_focal_stack_t(4:end-4,4:end-4,K), im_refocs(4:end-4,4:end-4,K))
end
%%
[~,im_refocs2] = VBM3D(im_refocs,3);
figure('Renderer', 'painters', 'Position', [600 600 500 200]);
[ha,pos] = tight_subplot(1,3,[.01 .01],[.05 .05],[.01 .01]);
for K_s =1:length(s_array2)
    axes(ha(K_s)); 
    imagesc(im_refocs2(2:end-1,2:end-1,K_s));
%     [~,idx]= min(abs(s_array2(K_s) - s_array))
%     imagesc(im_focal_stack_flt(:,:,idx));
%     title(['s: ' num2str(s_array(K_s))]);
    colormap('hot'); axis off; axis square;
end