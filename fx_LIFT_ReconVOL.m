function im_deconv = fx_LIFT_ReconVOL(Calib_Res, image_t, labmda_reg, options)
% ** takes the LIFT system information in [Calib_Res] and input measurement image [x,t]
% to reconstruct image [x,y,t] in streak mode operation
% Calib_Res: the system calibration results
% image_t:  the streak image
% labmda_reg: regularization hyperparameter
% ** output: a deconved image normalized into range [0,1] 

try
    gpuArray(1);
    canUseGPU = true;
    disp('GPU detected for acceleration...\n');
catch
    canUseGPU = false;
end

INVERT = options.INVERT;
CROP = options.CROP;
Deconv = options.Deconv;
if(Deconv)
    PSF = Calib_Res.PSF;
end
% shift_seats = Calib_Res.TD_shift_seats;
sub_img_cnt = round(Calib_Res.sub_img_cnt);
Reg_Size = Calib_Res.Reg_Size;

% overwrite new center for refocusing
if(options.Refocus)
    sub_img_cnt = options.sub_img_cnt;
end

angle = Calib_Res.Angle;
angle = (90 - angle);
N_angle = length(angle);
RESMAPLE = Calib_Res.RESAMPLE;

half_reg_size = round(Reg_Size/2);
N_img = half_reg_size*2;   % image size in each lenslet
crop_Imsize_half = round(N_img/sqrt(2)/2); % effective image region
cnt_img = round((N_img+1)/2); 

% ************
% zero padding the data to avoid out-of-range issue
% for the first and last lenslet region 
% ************
[nx_img,ny_img] = size(image_t);
image_t_pad = single(zeros(nx_img,ny_img +200));
image_t_pad(:,101:end-100) = image_t;
sub_img_cnt = sub_img_cnt + 100;
image_t = image_t_pad;
Nt = size(image_t,1);
image_t = norm1(image_t);

%% 3: setup solver
% setup cuda kernel for image rotation
k_cukernel = parallel.gpu.CUDAKernel('imrotate.ptx','imrotate.cu');
k_cukernel.GridSize = [ceil(N_img/32),N_img, Nt];
k_cukernel.ThreadBlockSize = [32, 1,1];   % reduce the block size if error raised (weaker GPU)

A = @(x) RF_CylArray_CUDA(x, single(angle), k_cukernel);
AT = @(x) RT_CylArray_CUDA(x, single(angle), N_img, Nt, k_cukernel);
if(canUseGPU)
    max_egival = power_iter_cube(A,AT,single(zeros(N_img,N_img,Nt,'gpuArray')));
else
    max_egival = power_iter_cube(A,AT,zeros(N_img,N_img,Nt));
end

opt.tol = 1*10^(-4);      % param for stopping criteria
opt.maxiter = 30;         % param for max iteration
opt.lambda = labmda_reg;  % param for regularizing param
opt.lambda_l1 = labmda_reg;
opt.vis = -1;

if(options.USE_TV)
    opt.denoiser = 'ProxTV';    % option of denoiser: BM3D, TV, ProxTV, VBM3D
else
    opt.denoiser = 'Prox_l1_GPU';    % option of denoiser: BM3D, TV, ProxTV, VBM3D
end

global GLOBAL_useGPU;
GLOBAL_useGPU = canUseGPU;
opt.step = 1*max_egival;    % step size
opt.monotone = 1;
opt.POScond = true;

% ************
% divide the slice into N_angle lenslet sub-regions
% ************
image_meas = zeros(Nt,N_img,N_angle);
for k = 1:N_angle
    if(RESMAPLE)
        half_reg_size = floor(0.5*Reg_Size./(cos(deg2rad(angle(k)))));
        if(mod(half_reg_size,2) == 1)
            half_reg_size = half_reg_size +1;
        end
        img_temp = image_t(:,sub_img_cnt(k)-half_reg_size:half_reg_size+sub_img_cnt(k)-1);
        image_meas(:,:,k) = resample(img_temp.', N_img,half_reg_size*2).';
    else
        half_reg_size = round(Reg_Size/2);
        image_meas(:,:,k) = image_t(:,sub_img_cnt(k)-half_reg_size:sub_img_cnt(k)+half_reg_size);
    end
end
image_meas = permute(image_meas,[2,3,1]); % [N_img,N_angle,Nt]
% b = reshape(image_meas,[N_img*N_angle*Nt,1]);
b = image_meas;

if(canUseGPU)
    x0 = single(zeros(N_img,N_img,Nt,'gpuArray')); 
    b = gpuArray(single(b));
else
    x0 = zeros(N_img,N_img,Nt);
end
%  call the solver
[x,convergence] = Solver_PlugPlay_FISTA2D(A,AT,b,x0,opt);
x = double(gather(x));

%% Postprocessing: deconv, invert and then crop
if(Deconv)
    PSF = PSF(20:end-20,20:end-20);
    for K = 1:size(x,3)
        x(:,:,K) = deconvlucy(squeeze(x(:,:,K)),PSF,10);    
    end
end

if(INVERT)
    x = rot90(x,2);
end

if(CROP)
    im_deconv = imcrop_local(x, cnt_img,crop_Imsize_half);
end
if(options.Normalize)
    im_deconv = norm1(im_deconv);
end
end