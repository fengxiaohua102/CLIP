%*********
% This script simulate the CLIP-LIFT image reconstrction
% using different transformation basis 
% 
clear all; clc; % close all;

% read raw orignial image
data_loader = load([pwd '\LightFieldData\clip_GFT_S1\CLIP_Data_LIFT.mat']);

angles = data_loader.angles; N_dec = 2; N_dec_ang = 1;
angles = angles(1:N_dec_ang:end,1:N_dec_ang:end,1:N_dec:end);
meas_data = data_loader.meas_data;  %[n,Na,Na,N_meas]
meas_data = meas_data(:,1:N_dec_ang:end,1:N_dec_ang:end,1:N_dec:end);
LF_data = data_loader.LF_sim_LIFT;
Nx = data_loader.Nx;
cnt = data_loader.cnt;
LFRes = size(LF_data);

[n,~,Na,N_meas] = size(meas_data);
% 3: process the measurement
meas_data = meas_data./max(meas_data(:));
% meas_data = meas_data + 0.2.*rand(size(meas_data));

% meas_data = gpuArray(single(meas_data));

%% SPC solver setup
% 1: model setup
opt_model.basis = 'ddct';
opt_model.wavelet_op = 'db4';  % wavelet type: db, sym, haar coif etc.
opt_model.s = -1.0;  % the refocusing parameter
opt_model.n = n;
opt_model.Nrefx = Na/2;
opt_model.Nrefy = Na/2;
if(strcmp(opt_model.basis, 'wavelet'))
    [a_flt,~,~,~] = wfilters(opt_model.wavelet_op);
    wavelet_len = length(a_flt)/2-1;
    n_size = ceil(n/2)+wavelet_len;
    opt_model.n_size = n_size;   
end
disp('calculating step size...');
A = @(x) LIFT_4DLF_F(x, angles, opt_model);
AT = @(x) LIFT_4DLF_T(x, angles, opt_model);

if(strcmp(opt_model.basis, 'wavelet'))
    max_egival = power_iter(A,AT,zeros(n_size,n_size*4))
else
    max_egival = power_iter(A,AT,zeros(n,n,Na,Na))
end

%% 2: setup solver
opt.maxiter =100;       % param for max iteration
opt.lambda = 5e-2;      % param for regularizing param 0.2e-1
opt.vis = -1;  
opt.denoiser = 'mixed';    % option of denoiser: BM3D, ProxTV
opt.POScond = false;       % positiveness contraint on the solution
global GLOBAL_useGPU;
GLOBAL_useGPU = 1;
opt.monotone = 1;
opt.step = 1.0*max_egival;  % step size

% 3: call solver
tic
x0 = zeros(n,n,Na,Na);
[x, convergence] = Solver_PlugPlay_FISTA2D(A, AT, meas_data, x0, opt);
x = reshape(x, [n,n,Na,Na]);
im_recon = norm1(x);
toc
LF_data_recon = im_recon( cnt-Nx/2:cnt+Nx/2-1, cnt-Nx/2:cnt+Nx/2-1,:,:);

figure;
subplot(2,2,1); imagesc(LF_data_recon(10:end-10,10:end-10,1,1)); colormap('hot');axis equal; axis off;title('FISTA');
subplot(2,2,2); plot(convergence); title('convergence'); axis square;

[ImgOut, FiltOptions, LF2] = LFFiltShiftSum( permute(LF_data_recon,[3,4,1,2]), opt_model.s);
ImgOut_norm = ImgOut(:,:,1);
subplot(2,2,3); imagesc(ImgOut_norm); colormap('hot'); axis equal; axis off;  title('LF refocus');
%%
s_array = linspace(-1,1,2)
im_focal_stack = norm1(LFrefocus( permute( LF_data_recon, [3,4,1,2]), s_array));
figure; montage(norm1(im_focal_stack(4:end-4,4:end-4,:))) ; colormap('hot'); title('focal stack');  %

%
LFRes = size(LF_data_recon);
LF_montage = reshape(LF_data_recon,[LFRes(1),LFRes(2),LFRes(3)*LFRes(4)]);
figure; montage(LF_montage, 'Size',[LFRes(3), LFRes(4)]) ; title('LF'); colormap('hot');
figure; montage(LF_montage(:,:,1:4:end), 'Size',[4, 4]) ; title('LF'); colormap('hot');

% ground truth light field refocus
im_focal_stack_t = norm1(LFrefocus( LF_data, s_array));
figure; montage(im_focal_stack_t) ; colormap('hot'); title('truth focal stack');  %norm1(im_focal_stack(4:end-4,4:end-4,:))

% calculate NMSE
[n_mse, ssimval] = eval_quality(im_focal_stack_t(4:end-4,4:end-4,2), im_focal_stack(4:end-4,4:end-4,2))

