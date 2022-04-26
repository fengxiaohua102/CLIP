%*********
% This script performs the CLIP-0D image reconstrction for the 4D light fields
% on experimental data 
%% read raw measurement image
clear all; clc; close all;
dataset = 'hada9'; %(4,4,16384)
filename = ['SPC_exp',dataset];
datapath = [pwd '\CLIP0D_data\' filename '\'];
REFOCUS = 1;  % flag to perform refocusing
Na = 4; N_meas = 256*1; Nx=128;

%%%%%%%
% meas_data_raw = load([datapath 'meas_data.mat']);  
% meas_data_raw = meas_data_raw.meas_data;%[Na,Na,N_meas]
meas_data_raw = zeros([Na,Na,Nx*Nx]);
for K = 1:Na
    for P = 1:Na
        data_loader = load([datapath 'big_data_raw' num2str(Na *(K-1) + P) '.mat']);
        meas_data_view = data_loader.big_data_raw;
        
        meas_data_view_mean = mean(meas_data_view([1,4098,8195,12292]));
        meas_data_view([1,4098,8195,12292])=[];
        meas_data_view = meas_data_view./meas_data_view_mean;
        
        meas_data_raw(K,P,:) = meas_data_view;
    end
end
codes_loader = load([pwd '\CLIP0D_data\ha128TVsort_16384.mat']);
% codes_loader = load([pwd '\hadamardcodes1\random128_16384.mat']);

codes_raw = single(codes_loader.codes);
clear codes_loader

%% extract CLIP data
meas_data = zeros(Na,Na,N_meas); codes = zeros(Nx,Nx,Na,Na,N_meas);
% idx_meas = randperm(Na*Na*N_meas);
idx_meas = 1:(Na*Na*N_meas);

ii = 1;
for i = 1:Na
    for j = 1:Na
        st = ((i-1)*Na+(j-1))+1; 
        idx = 1+N_meas*(ii-1):N_meas*ii ; % st:Na^2:Na*Na*N_meas;
         meas_data(i,j,:) = meas_data_raw(i,j, idx_meas(idx));
         codes(:,:,i,j,:) = codes_raw(:,:,1,1, idx_meas(idx));
%          ii = ii+1;% mod(ii+1,2)+1;
    end
end

[m,n,~,Na,N_meas] = size(codes);

%% preprocess the codes and measurement data
% 3: process the measuremen
% **** New step that correct data acquisition flaw ***
[meas_data, outlier_idx] = rm_meas_data_outlier(meas_data,false);
meas_data(outlier_idx) = 0;

codes_temp = reshape(codes, [m*n,Na*Na*N_meas]);
for K  = 1: size(codes_temp,1)
    codes_temp(K,~outlier_idx) = codes_temp(K,~outlier_idx) - mean(codes_temp(K,~outlier_idx) );
    codes_temp(K,outlier_idx) = 0;  % extra correction step for the codes
end

fac = 1.0/mean(vecnorm(codes_temp(:,~outlier_idx),2,2));
codes = reshape(codes_temp.*fac, [m,n,Na,Na,N_meas]); 

% codes = gpuArray(single(codes));
% meas_data = gpuArray(single(meas_data));
% codes = gather(codes); meas_data = gather(meas_data);
%% SPC solver setup
% 1: model setup
opt_model.basis = 'ddct';
opt_model.wavelet_op = 'sym4';  % wavelet type: db, sym, haar coif etc.
opt_model.s = -1;  % the refocusing parameter
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
max_egival = power_iter(A,AT,zeros(m,n,Na,Na))

%% 3.FSITA recon 
opt.maxiter = 50;         % param for max iteration
opt.lambda = 3e-1;      % param for regularizing param 5e-2
opt.vis = -1;
opt.denoiser = 'mixed';   % option of denoiser: BM3D, ProxTV ProxTV_Med
opt.POScond = false;       % positiveness contraint on the solution
global GLOBAL_useGPU;
GLOBAL_useGPU = 1;
opt.monotone = 1;
opt.step = 1.0*max_egival;  % step size

x0 = zeros(m,n,Na,Na);
tic
[LF_recon,convergence] = Solver_PlugPlay_FISTA2D_SPC(A,AT,reshape(meas_data,Na,Na,N_meas),x0,opt);
toc
% LF_recon = mirt_idctn(LF_recon);
%
figure;
subplot(2,2,1); imagesc(LF_recon(10:end-10,10:end-10,1,1)); colormap('hot');axis equal; axis off;title('FISTA');
subplot(2,2,2); plot(convergence); title('convergence'); axis square;

[ImgOut, FiltOptions, LF2] = LFFiltShiftSum( permute(LF_recon,[3,4,1,2]), 0.5);
ImgOut_norm = ImgOut(:,:,1);
subplot(2,2,3); imagesc(ImgOut_norm); colormap('hot'); axis equal; axis off;
LFRes = size(LF_recon);
figure; montage(norm1(reshape(LF_recon,[LFRes(1),LFRes(2),LFRes(3)*LFRes(4)])), 'Size',[LFRes(3), LFRes(4)]) ;
colormap('hot'); title('CLIP 4DLF');

s_array = linspace(-2,2,4); 
im_focal_stack = norm1(LFrefocus( permute(LF_recon,[3,4,1,2]), s_array));
figure; montage((im_focal_stack)); colormap('hot');axis square; axis off;title('CLIP');

%% canonical light field refocusing (Light data reconstructed)
if(REFOCUS)
    LF_data = load([datapath 'matlab64view' , '.mat']);
    LF_data = LF_data.im_recon_LF;
    [m,n,Na,~] =size(LF_data);
    % **** visulize the light fields
    LF_img = reshape(LF_data,[m,n,Na*Na]);
    for K = 1:size(LF_img,3)
        LF_img(:,:,K) = norm1(LF_img(:,:,K));
    end
    figure; montage(norm1(LF_img));colormap('hot'); axis square; title('light fields');
%     im_focal_stack_gt = zeros(m,n,length(s_array));
%     for K= 1:length(s_array)
%         [ImgOut, FiltOptions, LF2] = LFFiltShiftSum( permute(LF_data,[3,4,1,2]), s_array(K)*0.5);
%         ImgOut_norm = ImgOut(:,:,1);
%         im_focal_stack_gt(:,:,K) = norm1(ImgOut_norm);
%     end
    im_focal_stack_gt = norm1(LFrefocus( permute(LF_data,[3,4,1,2]),s_array));
    figure; montage((im_focal_stack_gt)); colormap('hot');axis square; axis off;title('Ground truth');
end
%%
[n_mse, ~] = eval_quality(LF_recon,LF_data)