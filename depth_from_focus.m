function [depth_fit, depth_map, Img_ALLFOCUS] = depth_from_focus(Img_Stack, Method, thresh, FIT, VIS)
% ******
% perform depth from focus 
% Img_Stack is the refocused image at different depths [Nx,Ny,Ndepth]
% thresh: the thresholding that mask out low intensity region
% NOTE: the Img_Stack shoud be aligned!
% ******

[Nx,Ny,N_STEP] = size(Img_Stack);
ImgFM_Stack = zeros(Nx,Ny,N_STEP);
% 1: compute the foucs measure at corresponding depths [Nx,Ny, Ndepth]
for K = 1:N_STEP
    im_fm = FocusMeasure(Img_Stack(:,:,K),Method);
%     im_fm = medfilt2(im_fm, [7,7]);
    ImgFM_Stack(:,:,K) = im_fm;
end

Img_FM_sort = sort(ImgFM_Stack,3,'descend');
% computing the confidence: not used right now
Img_CM = (Img_FM_sort(:,:,1) - Img_FM_sort(:,:,2))./sum(Img_FM_sort(:,:,1),3);

% 2: Select the best depth/slop based on the best focus-measure
% first form the all-in-focus image by selecting the best-in-focus stack index
[~, idx] = max(ImgFM_Stack,[],3);

idx = prox_tv1d(double(idx)./N_STEP,1.0);
idx = prox_tv1d(idx.',1.0).';
idx = round(idx.*N_STEP);
idx(idx==0)=1;
Img_Org_Reshape = reshape(Img_Stack, Nx*Ny,N_STEP);
Img_ALLFOCUS = zeros(Nx*Ny,1);

for K = 1: size(Img_ALLFOCUS,1)
    Img_ALLFOCUS(K) = Img_Org_Reshape(K,idx(K));
end
Img_ALLFOCUS = reshape(Img_ALLFOCUS,Nx,Ny);
% mask = (Img_ALLFOCUS<thresh*max(Img_ALLFOCUS(:)));

% depth_map =  medfilt2(idx,[11,11]);
depth_map = idx;
% depth_map(mask) = NaN;

%% try Gaussian fitting of each pixel that show meaningful depth
if(FIT)
    Img_FM_Fit = zeros(5,1); % use 5 points around maximum idx for fitting
    depth_fit = zeros(Nx*Ny,1);
    Img_FM_Reshaped = reshape(ImgFM_Stack,Nx*Ny,[]);
    gaussEqn = 'a*exp(-((x-b)/c)^2) +d';
    tic
    for K = 1 : Nx*Ny
        if(~mask(K))
            SKIP = false;
            idx_fit = zeros(5,1);
            for P = 1:5
                try
                    Img_FM_Fit(P) = Img_FM_Reshaped(K,idx(K)+P-3);
                    idx_fit(P) = idx(K)+P-3;                
                catch
                    % disp('Error detected, skipping ...')
                    SKIP = true;
                    continue;
                end

            end
            if(~SKIP)
                startPoints = [Img_FM_Fit(3) idx_fit(3) 2 min(Img_FM_Fit(:))];
                % Gaussian fitting
                p = fit(idx_fit, Img_FM_Fit, gaussEqn,'Start', startPoints);  
                coeff = coeffvalues(p);
                if(coeff(2)>0)
                    depth_fit(K) = coeff(2);
                else
                    depth_fit(K) = depth_map(K);
                end
            else
                depth_fit(K) = depth_map(K);
            end
        end
    end
    toc
    depth_fit = reshape(depth_fit,[Nx,Ny]);
%     depth_fit = medfilt2(depth_fit,[7,7]);
else
    depth_fit = depth_map;
end

% fun = @(x) mode(x(:));
% depth_fit = nlfilter(depth_fit, [7,7], fun);

if(VIS)
    figure; subplot(2,2,1); imagesc(depth_map);colormap('hot'); title('Depth map');
    subplot(2,2,2); imagesc(Img_CM);colormap('hot'); title('Confidence map');
    subplot(2,2,3); imagesc(Img_ALLFOCUS); title('ALl-in-focus image');
    subplot(2,2,4); imagesc(depth_fit);colormap('hot'); title('Fit depth map');
%     surf(depth_fit,'LineStyle','None'); colormap('hot'); title('Fit depth surface');
end

end