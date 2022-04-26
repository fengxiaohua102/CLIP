function im_r = imrotate_gpu(k,im, out,angle_r)
% k: the cuda kernel object in matlab
% im: input 3D
% out: output 3D
    [nx,ny,~] = size(im);
    im_r = feval(k,out,im, nx,ny, angle_r);
end