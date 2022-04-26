%*********
% This script performs the SPC image reconstrction
% on experimental data 
%% read raw measurement image
clear all; clc; %close all;
% clear codes meas_data
myy = 'hada16'; %(4,4,16384)
filename = ['SPC_exp',myy];
datapath = [pwd '\SPC_data\' filename '\'];
REFOCUS = 0;  % flag to perform refocusing
N_dec = 1;
Na = 4; N_meas = 1024/N_dec; Nx=128;

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
codes_loader = load([pwd '\hadamardcodes1\ha128TVsort_16384.mat']);
% codes_loader = load([pwd '\hadamardcodes1\random128_16384.mat']); 
codes_raw = single(codes_loader.codes);
clear codes_loader

%% extract CLIP data
meas_data = zeros(Na,Na,N_meas); codes = zeros(Nx,Nx,Na,Na,N_meas);
% idx_meas = randperm(Na*Na*N_meas);
idx_meas = 1:(Na*Na*N_meas);

ii = 1; ii2 = 0;
for i = 1:Na
    for j = 1:Na
        st = ((i-1)*Na+(j-1))+1; 
        ii = mod(ii2,1)+1 ; % ii+1; 
        idx = 1+N_meas*(ii-1):N_meas*ii ; % st:Na^2:Na*Na*N_meas;
         meas_data(i,j,:) = meas_data_raw(i,j, idx_meas(idx) );
         codes(:,:,i,j,:) = codes_raw(:,:,1,1, idx_meas(idx) );
         ii2 = ii2+1;         
    end
end

[m,n,~,Na,N_meas] = size(codes);


%% preprocess the codes and measurement data
codes_temp = reshape(codes, [m*n,Na*Na*N_meas]);
for K  = 1: size(codes_temp,1)
    codes_temp(K,:) = codes_temp(K,:) - mean(codes_temp(K,:) );
end

fac = 1.0/mean(vecnorm(codes_temp,2,2));
codes = reshape(codes_temp.*fac, [m,n,Na,Na,N_meas]); 

% codes = gpuArray(single(codes));
% meas_data = gpuArray(single(meas_data));
% codes = gather(codes); meas_data = gather(meas_data);
%% SPC solver setup
% add error reading to the measurement data
meas_data2 = meas_data./max(meas_data(:));
n_err = floor(0.005.*length(meas_data(:)));
idx = randsample(length(meas_data(:)), n_err);
idx2 = randsample(length(meas_data(:)), n_err);
meas_data2(idx) = 0;
meas_data2(idx2) = 1;
figure; imagesc(reshape(meas_data2,128,128));
% meas_data2 = meas_data;

% 3: process the measurement
% **** New step that correct data acquisition flaw ***
[meas_data2, outlier_idx] = rm_meas_data_outlier(meas_data2,false);
% figure;plot(meas_data(4097:4096*2),'*-'); sum(outlier_idx)
meas_data2(outlier_idx) = 0;

% 1: model setup
opt_model.basis = 'dwavelet';
opt_model.wavelet_op = 'sym4';  % wavelet type: db, sym, haar coif etc.
opt_model.s = -0;  % the refocusing parameter
opt_model.Nrefx = Na/2;
opt_model.Nrefy = Na/2;
if(strcmp(opt_model.basis, 'wavelet'))
    [a_flt,~,~,~] = wfilters(opt_model.wavelet_op);
    wavelet_len = length(a_flt)/2-1;
    n_size = ceil(n/2)+wavelet_len;
    opt_model.n_size = n_size;   
end

A = @(x) CLIP_F_ACC(x, codes, opt_model);
AT = @(x) CLIP_T_ACC(x, codes, opt_model);

% 2.reconstruction using the DAMP algorithm
% im_recon2 = DAMP(double(meas_data),50,m,n,'Med2D',A,AT);  % iter number
% figure; imagesc(im_recon2); colormap('hot'); axis square; axis off; title('DAMP');
max_egival = power_iter(A,AT,zeros(m,n))
%% 3.FSITA recon 
opt.maxiter = 40;      % param for max iteration
opt.lambda = 8e0;      % param for regularizing param 5e-2 6e0*max_egival
opt.vis = -1;
opt.denoiser = 'mixed';  % option of denoiser: BM3D, ProxTV,ProxTV_Med
opt.POScond = 1;       % positiveness contraint on the solution
global GLOBAL_useGPU;
GLOBAL_useGPU = 1;
opt.monotone = 1;
opt.step = 1.0*max_egival;  % step size
%
x0 = zeros(m,n);
[recon_now,convergence] = Solver_PlugPlay_FISTA2D(A,AT,double(reshape(meas_data2,[Na,Na,N_meas])),x0,opt);
% [LF_recon,convergence] = Solver_RED_APG(A,AT,reshape(meas_data,Na,Na,N_meas),x0,opt);

recon_now = reshape((recon_now),[m,n]);
% recon_now_flt = medfilt2(recon_now,[5,5]);
% recon_now = DAMP(double(meas_data)*1,30,m,n,'Med2D',A,AT);
figure; subplot(1,2,1); imagesc(recon_now); axis square; colormap(hot); 
subplot(1,2,2); plot(convergence);

%% produce a focal stack
REFOCUS = 0; opt.lambda =15e-0;      % param for regularizing param 5e-2 
opt.denoiser = 'mixed';
if(REFOCUS)
    s_array = linspace(-1,1,4).*2; %-0.7;
    im_focal_stack = zeros(m,n,length(s_array));
   % tic
    for K= 1:length(s_array)
        tic
        opt_model.s = s_array(K);
        A = @(x) CLIP_F_ACC(x, codes, opt_model);
        AT = @(x) CLIP_T_ACC(x, codes, opt_model);
%         recon_now = DAMP(double(meas_data)*1,50,m,n,'Med2D',A,AT);
        [recon_now,convergence] = Solver_PlugPlay_FISTA2D(A,AT,double(reshape(meas_data,[Na,Na,N_meas])),x0,opt);
        recon_now = reshape((recon_now),[m,n]);
        im_focal_stack(:,:,K) = norm1(recon_now);
%         figure; subplot 121;imagesc(recon_now(10:end-10,10:end-10),[0.002 0.02]); axis square; colormap(hot); 
%         subplot(1,2,2); plot(convergence);
        toc
    end
    %toc
    figure; montage((im_focal_stack),  'Size',[1,length(s_array)]); colormap('hot'); axis off;title('CLIP');
    %save([datapath 'matlabrefocusclip' , '.mat'], 'im_focal_stack','im_recon2');
end

%% canonical light field refocusing (Light data reconstructed)
if(REFOCUS)
    LF_data = load([datapath 'matlab64view' , '.mat']);
    LF_data = LF_data.im_recon_LF;
    [m,n,Na,~] =size(LF_data);
    [vv, uu] = ndgrid(1:m, 1:n);
    Nrefx = Na/2; Nrefy = Na/2;
    im_refocus = zeros(m,n,length(s_array));
    for L = 1:length(s_array)
        s = s_array(L);
        for K_x = 1:Na
            nx = (s*(K_x-Nrefx));
            for K_y = 1:Na
                % 1: apply shearing operator             
                ny = (s*(K_y-Nrefy));
                % image_sheared = my_shift(image_object,nx,ny);  
                % 1b: interpolation
                image_sheared = interp2(squeeze(LF_data(:,:,K_x,K_y)), uu+ny, vv+nx, 'cubic',0);
                im_refocus(:,:,L) = im_refocus(:,:,L) + image_sheared;% - mean(mean(image_sheared(110:end-10,13:end-10,:)));
            end
        end
        im_refocus(:,:,L) = norm1( im_refocus(:,:,L) ) ;
    end
    figure; montage(im_refocus,'Size',[1,length(s_array)]); colormap('hot'); axis off; title('refocused images');
    % **** visulize the light fields
    LF_img = reshape(LF_data,[128,128,Na*Na]);
    for K = 1:size(LF_img,3)
        LF_img(:,:,K) = norm1(LF_img(:,:,K));
    end
    figure; montage(norm1(LF_img));colormap('hot'); axis square; title('light fields');
end
%% calculate NMSE
% for L = 1:length(s_array)
%     [n_mse, ~] = eval_quality(im_refocus(:,:,L),im_focal_stack(:,:,L))
% end
