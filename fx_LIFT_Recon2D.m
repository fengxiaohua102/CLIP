function [im , image_meas] = fx_LIFT_Recon2D(Calib_Res, image_line, lambda_reg, options)
% takes the UTP system information [Calib_Res] and input measurement 1D image 
% to reconstruct image [x,y] in focus mode operation
% Calib_Res: the system calibration results
% image_line:  the 1D image
% labmda_TV: TV regularization hyperparameter
% ** output: a deconved image

try
    gpuArray(1);
    canUseGPU = true;
    disp('GPU detected for acceleration...\n');
catch
    canUseGPU = false;
end

INVERT = options.INVERT;
CROP = options.CROP;
USE_TV = options.USE_TV;

% provide the option to refocus
if(options.Refocus)
    sub_img_cnt = options.sub_img_cnt;
else
    sub_img_cnt = round(Calib_Res.sub_img_cnt);
end
Reg_Size = Calib_Res.Reg_Size;

angle = Calib_Res.Angle;
angle = 90 - angle;
N_angle = length(angle);
half_reg_size = round(Reg_Size/2);
N_img = half_reg_size*2;                   % image size in each lenslet
image_meas = zeros(N_img,N_angle);
crop_Imsize_half = round(N_img/sqrt(2)/2); % effective image regoin size
cnt_img = round((N_img+1)/2);
RESMAPLE = Calib_Res.RESAMPLE;

image_line = norm1(image_line); 

% zero padding the data to avoid out-of-range issue for the first and last
% lenslet region
nx_img = length(image_line);
image_line_pad = zeros(1,nx_img+100);
image_line_pad(51:end-50) = image_line;
sub_img_cnt = sub_img_cnt + 50;
image_line = image_line_pad;

% 3: divide the slice into N_angle lenslet sub-regions
for k = 1:N_angle
    k
    if(RESMAPLE)
        half_reg_size = floor(0.5*Reg_Size./(cos(deg2rad(angle(k)))));
        if(mod(half_reg_size,2) == 1)
            half_reg_size = half_reg_size +1;
        end
        img_temp = image_line(sub_img_cnt(k)-half_reg_size:half_reg_size+sub_img_cnt(k)-1);
        image_meas(:,k) = resample(img_temp,N_img,half_reg_size*2);
    else
        half_reg_size = round(Reg_Size/2);
        image_meas(:,k) = image_line(sub_img_cnt(k)-half_reg_size:sub_img_cnt(k)+half_reg_size-1);
    end
end
b = image_meas;

%% Number of FISTA main loop iterations
A = @(x) RF_CylArray(x,angle);
AT = @(x) RT_CylArray(x,angle,N_img,1);
if(canUseGPU)
    max_egival = power_iter(A,AT,single(zeros(N_img,N_img,'gpuArray')));
else
    max_egival = power_iter(A,AT,zeros(N_img,N_img));
end

opt.tol = 1*10^(-5);  % param for stopping criteria
opt.maxiter = 50;     % param for max iteration
opt.lambda = lambda_reg;     % param for regularizing param
opt.vis = -1;
if(USE_TV)
    opt.denoiser = 'ProxTV';  % option of denoiser: BM3D, TV, ProxTV, VBM4D
else
    opt.denoiser = 'Prox_l1_GPU';
end
global GLOBAL_useGPU;
GLOBAL_useGPU = 1;
opt.step = 1*max_egival;    % regularizing parameter
opt.monotone = 1;
opt.POScond = true;

if(canUseGPU)
    x0 = single(zeros(N_img,N_img,'gpuArray')); 
    b = gpuArray(single(b));
else
    x0 = zeros(N_img,N_img);
end
[im,convergence] = Solver_PlugPlay_FISTA2D(A,AT,b,x0,opt);
im = double(gather(im));
% deconv, invert and then crop
if(options.DECONV)  
    PSF = Calib_Res.PSF;
	im = deconvlucy(im,PSF,10);
end
if(INVERT)
    im = rot90(im,2);
end
if(CROP)
    im = imcrop_local(im, cnt_img, crop_Imsize_half);
end
end