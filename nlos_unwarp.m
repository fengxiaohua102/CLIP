function [nlos_data]= nlos_unwarp(im_crop,nlos_geo, dt)
% unwarp the data using the NLOS system geometry
% im_cube: the 3D [x,y,t] LIFT reconstruction data
% nlos_geo: nlos geometry calibration results
% dt: the sampling time resolution of the 3D data cube

DeltaT  = 3e8 * dt;
nlos_data = im_crop;  % must be in shape [Nx, Ny, Nt]

% grid_pos: the 2D grid position on the wall
% cam_pos: the LIFT camera position in the global coo.
% laser_pos : laser spot on the wall
camera_pos = nlos_geo.camera_pos;  % the LIFT camera pos on global coo.
laser_pos = nlos_geo.laser_pos;  % laser pos on the wall, relative to global coo.
grid_pos = nlos_geo.grid_pos;  % detection pos on the wall, relative to global coo.
% calculate r2’-r2 = abs(grid_pos – cam_pos)-abs(laser_pos - cam_pos)  
laser_2_cam = vecnorm(laser_pos - camera_pos)
for K = 1:size(nlos_data,1)
    for P = 1: size(nlos_data,2)
        grid_2_cam = vecnorm(squeeze(grid_pos(:,K,P)) - camera_pos');
        delay_t = round((grid_2_cam - laser_2_cam)/DeltaT);
        if(isnan(delay_t))
            delay_t = 0;
        end
        nlos_data(K,P,:) = circshift(nlos_data(K,P,:), -delay_t);
    end
end 
end    

