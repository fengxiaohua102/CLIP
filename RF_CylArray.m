function image_F = RF_CylArray(image_cube,angle)
    % This is the accelerated version
    % implement the forward operator that of tomographic photography
    % angle is the two d array specifying the angle in degrees of integration in each lenslet
    % image_cube: [n,n,Nt]
    % image_F: a vector of [n*N_angle*Nt(Nt),1]
    
    GPU_ACC = isa(image_cube, 'gpuArray');
    N_angle = length(angle);
    [m,n,Nt] = size(image_cube);
    
    % generate mask to account for the rotation operation that may
    % generate shapes larger than original image
    x = ((1:n)-n/2)/n;     y = x;
    [x,y] = meshgrid(x,y);
    mask = (x.^2+y.^2)>=0.25;
    mask = repmat(mask,[1,1,Nt]);
    
    if(GPU_ACC)
        line_image_t = single(zeros(n,Nt,N_angle,'gpuArray'));
        mask = gpuArray(mask);
    else
        line_image_t = zeros(n,Nt,N_angle);
    end
    image_cube(mask) = 0;         
    for K = 1:N_angle
        im = imrotate(image_cube,angle(K),'bilinear', 'crop');
        line_image_t(:,:,K) = sum(im,1); 
    end
    line_image_t = permute(line_image_t,[1,3,2]);
    % line_image shape: [n,N_angle,Nt]
    image_F = line_image_t;

end