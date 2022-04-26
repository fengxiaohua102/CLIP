%****** performs 3D NLOS reconstruction using CUDA
clear all; clc; close all;

top_DIR = [pwd '\NLOS\'] 

% load the LIFT reconstructed results
mat_content = load([top_DIR 'y4_iradon.mat']);  % _new
sensor_data = mat_content.nlos_data;
sensor_data = (single(sensor_data));
sensor_data = permute(sensor_data,[3,2,1]);
[N_t,N_X,N_Y] = size(sensor_data);
fprintf('Original sensor_data shape: %d, %d, %d\n', N_X, N_Y, N_t);
N_pos = int32(N_X*N_Y);
N_t = int32(N_t);

% phasor field parameters: ('Manq' use [10,5] and dx = 5e-3 for best resutls)
lambda_wave = 8; N_cycle = 3;
ReceiveDelay = 50;
c_dt = single(mat_content.DeltaT);        % (temporal_resolution)*c
Laserspot = gpuArray( single(mat_content.laser_grid_pos));    % laser spot position
sensor_pos = single(mat_content.camera_grid_pos) ;   % detection spot position on the wall
sensor_pos = gpuArray( single(sensor_pos));  
sensor_pos = permute(sensor_pos,[3,2,1]);  % must be in shape [Nx,Ny,3]

%%
% *********
% 2: Setup the reconstruction parameters
% *********
% reconstruction cube size (together with dx)
N_RECON = 128;  %% 100
N_Reconz = 128; %% 100
NY = int32(N_RECON);
NZ = int32(N_Reconz);   
    
dx = single(8e-3);  % reconstruction grid size in x-y plane
wave_dx = double(dx); 
dz = dx ;           % reconstruction grid size in z (depth) direction
% setup the local origin in global coo. for the reconstruction cube
origin = gpuArray( single([-0*dx, 0*dx, -0*dz]) );  
% X(increase.FOV toward-left-shift), Y(increase.FOV to
NX = int32(N_RECON);

%***** set the cudakernal object for GPU calculation
% a. Create CUDAKernel object.
k = parallel.gpu.CUDAKernel('PhasorField.ptx','PhasorField.cu');
% b. Set object properties.
k.GridSize = [ceil(NX/32),NY,NZ];
k.ThreadBlockSize = [32, 1, 1];
% c. Call feval with defined inputs 
Recon_out = single(zeros(NX,NY,NZ, 'gpuArray')); % Input gpuArray.
% generate phasor data
[RF_data_real_array, RF_data_imag_array] = Generate_phasor_data(sensor_data, lambda_wave*dx, N_cycle, c_dt);

tic
[NLOS_Vol] = feval(k,Recon_out,RF_data_real_array, RF_data_imag_array, sensor_pos, origin, Laserspot, ...
    dx, dz, NX,NY,NZ, c_dt, N_pos, N_t, ReceiveDelay);
NLOS_Vol = gather(NLOS_Vol);
recon_time = toc

%%
NLOS_Vol = NLOS_Vol./max(NLOS_Vol(:));
im_MPI = squeeze(max(NLOS_Vol,[],3)).';
im_MPI = medfilt2(im_MPI,[3,3]);
% im_MPI(end - 10, end - 30: end - 10)= 1;

figure; imagesc(im_MPI,[0.05,max(im_MPI(:))]);  title( num2str(ReceiveDelay));colormap('hot'); axis square;axis off;
% Write the reconstructed 3D in vtk format for visualization in Paraview.
% vtkwrite([DIR filename '_NLOS_Recon_iradon.vtk'], 'structured_points', 'NLOS_Vol', NLOS_Vol);
%% 
% figure; imagesc(img_set(:,:,6),[0.1,0.8]); colormap('hot'); axis square;axis off

%  figure;
%  for i = 1:100
%      imagesc(NLOS_Vol(:,:,i));
%      pause(0.1);
%  end
 
% map = zeros(256,3);
% green_linear = (0:255).'; map(:,2) = green_linear./255;
% psf_d = psf(:,:,20); psf_d = imgaussfilt(psf_d,0.75);
% figure; imagesc(psf_d); colormap(map);