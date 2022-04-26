%*********
% This script simulate the CLIP-LIFT image reconstrction
% using different transformation basis and test the robustness of CLIP-1D
% 
clear all; clc; close all;

%% read raw orignial image
% data_loader = load([pwd '\scene6_LF_Data_LIFT.mat']);
data_loader = load([pwd '\LightFieldData\clip_GFT_S2\CLIP_Data_LIFT.mat']);

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
% Add Possion noises
N_photons = 10000;
meas_data = uint16(meas_data./max(meas_data(:)) .* N_photons);
meas_data_n = imnoise(meas_data, 'poisson');
figure; plot(meas_data_n(:),'-r'); hold on; plot(meas_data(:),'--k');

meas_data = double(meas_data_n)./N_photons;

% meas_data = gpuArray(single(meas_data));

%% SPC solver setup
% 1: model setup
opt_model.basis = 'ddct';
opt_model.wavelet_op = 'db4';  % wavelet type: db, sym, haar coif etc.
opt_model.s = 0.0;  % the refocusing parameter
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
A = @(x) LIFT_F_Transf(x, angles, opt_model);
AT = @(x) LIFT_T_Transf(x, angles, opt_model);

if(strcmp(opt_model.basis, 'wavelet'))
    max_egival = power_iter(A,AT,zeros(n_size,n_size*4))
else
    max_egival = power_iter(A,AT,zeros(n,n))
end
 
%% 2: setup solver
opt.maxiter = 100;       % param for max iteration
opt.lambda = 100e0;      % param for regularizing param [200e0, 100e0, 100e0] for S2 and [10e0, 10e0, 50e0]
opt.vis = -1;  
opt.denoiser = 'mixed';   % option of denoiser: BM3D, ProxTV
opt.POScond = false;       % positiveness contraint on the solution
global GLOBAL_useGPU;
GLOBAL_useGPU = 1;
opt.monotone = 1;
opt.step = 1.0*max_egival;  % step size

% 3: call solver
tic
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
        x0 = zeros(n,n);
        [x, convergence] = Solver_PlugPlay_FISTA2D(A, AT, meas_data, x0, opt);
        x = reshape(x, [n,n]);
        im_recon = norm1(x);
end
toc
im_recon = im_recon( cnt-Nx/2:cnt+Nx/2-1, cnt-Nx/2:cnt+Nx/2-1);

%%
figure;
subplot(2,2,1); imagesc(im_recon); colormap('hot');axis equal; axis off;title('FISTA');
subplot(2,2,2); plot(convergence); title('convergence'); axis square;

[ImgOut, FiltOptions, LF2] = LFFiltShiftSum( LF_data, opt_model.s);
Img_ref = ImgOut(:,:,1);
subplot(2,2,3); imagesc(Img_ref); colormap('hot'); axis equal; axis off; title('ground truth');
subplot(2,2,4); imagesc(squeeze(LF_data(4,4,:,:))); colormap('hot'); axis equal; axis off; title('Central view');

% calculate NMSE
[n_mse, ssimval] = eval_quality(Img_ref, im_recon)
