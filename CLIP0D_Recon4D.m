%*********
% This script simulate the SPC image reconstrction
% using different transformation basis 
%% read raw measurement image
clear all; clc; %close all;
% data_loader = load([pwd '\scene6_LF_Data_SPC.mat']);
data_loader = load([pwd '\LightField\clip_GFT_S1\CLIP_Data_SPC.mat']);

codes = data_loader.codes; N_dec = 2;
codes = codes(:,:,:,:,1:N_dec:end);
meas_data = data_loader.meas_data;  %[Na,Na,N_meas]
meas_data = double(meas_data(:,:,1:N_dec:end));

LF_data = data_loader.LF_sim_SPC;
[m,n,~,Na,N_meas] = size(codes);

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

% codes = gpuArray(single(codes));
% meas_data = gpuArray(single(meas_data));

%% SPC solver setup
% 1: model setup
opt_model.basis = 'dwavelet';
opt_model.wavelet_op = 'sym4';  % wavelet type: db, sym, haar coif etc.
opt_model.s = -1.0;  % the refocusing parameter
opt_model.Nrefx = Na/2;
opt_model.Nrefy = Na/2;
if(strcmp(opt_model.basis, 'wavelet'))
    [a_flt,~,~,~] = wfilters(opt_model.wavelet_op);
    wavelet_len = length(a_flt)/2-1;
    n_size = ceil(n/2)+wavelet_len;
    opt_model.n_size = n_size;   
end

A = @(x) SPC_4DLF_F_ACC(x, codes, opt_model);
AT = @(x) SPC_4DLF_T_ACC(x, codes, opt_model);

    
%% 2: setup solver with FISTA
disp('calculating step size...');
max_egival = power_iter(A,AT,zeros(m,n,Na,Na))

%% 3.FSITA recon 
opt.maxiter = 100;         % param for max iteration
opt.lambda = 0.1e-4;      % param for regularizing param 5e-2
opt.vis = -1;
opt.denoiser = 'ProxTV_Med';   % option of denoiser: BM3D, ProxTV
opt.POScond = false;       % positiveness contraint on the solution
global GLOBAL_useGPU;
GLOBAL_useGPU = 1;
opt.monotone = 1;
opt.step = 1.0*max_egival;  % step size

x0 = zeros(m,n,Na,Na);
tic
[LF_recon,convergence] = Solver_PlugPlay_FISTA2D(A,AT,reshape(meas_data,Na,Na,N_meas),x0,opt);
% [LF_recon,convergence] = Solver_PlugPlay_ADMM(A,AT,reshape(meas_data,Na,Na,N_meas),x0,opt);
% [LF_recon,convergence] = Solver_RED_APG(A,AT,reshape(meas_data,Na,Na,N_meas),x0,opt);

toc
%
figure;
subplot(2,2,1); imagesc(LF_recon(:,:,1,1)); colormap('hot');axis equal; axis off;title('FISTA-view');
subplot(2,2,2); plot(convergence); title('convergence'); axis square;
[ImgOut, FiltOptions, LF2] = LFFiltShiftSum( permute(LF_recon,[3,4,1,2]), 0.5);
ImgOut_norm = ImgOut(:,:,1);
subplot(2,2,3); imagesc(ImgOut_norm); colormap('hot'); axis equal; axis off;
LFRes = size(LF_recon);
LF_filt = reshape(LF_recon,[LFRes(1),LFRes(2),LFRes(3)*LFRes(4)]);
% for K = 1:size(LF_filt,3)
%     LF_filt(:,:,K) = norm1( medfilt2(LF_filt(:,:,K), [5,5] ));
% end
LF_montage = permute( reshape(LF_filt,[n,n,Na,Na]), [3,4,1,2]);
[ImgOut, FiltOptions, LF2] = LFFiltShiftSum(LF_montage, 0.5);
ImgOut_norm = ImgOut(:,:,1);
subplot(2,2,4); imagesc(ImgOut_norm); colormap('hot'); axis equal; axis off;

figure; montage(LF_filt, 'Size',[LFRes(3), LFRes(4)]) ; colormap('hot');

s_array = linspace(-1,1,2); 
im_focal_stack = norm1(LFrefocus(LF_montage , s_array));
figure; montage(im_focal_stack) ; colormap('hot');
figure; montage(LF_filt(:,:,1:4:end), 'Size',[4, 4]) ; title('LF'); colormap('hot');

% ground truth light field refocus
im_focal_stack_t = norm1(LFrefocus( LF_data, s_array));
figure; montage(im_focal_stack_t) ; colormap('hot'); title('truth focal stack');  %norm1(im_focal_stack(4:end-4,4:end-4,:))

% calculate NMSE
[n_mse, ssimval] = eval_quality(im_focal_stack_t(4:end-4,4:end-4,2), im_focal_stack(4:end-4,4:end-4,2))