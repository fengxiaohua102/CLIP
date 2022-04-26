function [P_real, P_imag] = Generate_phasor_data(dataCube, lambda, N_cycle, dt)
% It accepts the 3D data cube and generate a complex phasor pulse
% from captured temporal measurements.
%   * origial code from Xiaochun Liu et. al.

%   P = Generate_phasor_data(lambda, times, delta_t, H)
%
% parameter description:
%   dataCube: 3D temporal measurements [x,y,t]
%   lambda:  phasor field wavelength (m)
%   N_cycle:  number of cycles inside the pulse 
%   dt: time resolution of the datacube

    % Time bins covered by the phasor pulse
    pulse_size = round((N_cycle * lambda) / dt);
    
    % Time bins covered by a single cycle
    cycle_size = round(lambda / dt);
    
    % Select a sigma so that the 99% of the gaussian is inside
    % the pulse limits
    sigma = (N_cycle * lambda) / 6;
    
    % Virtual emitter emission profile
    t = dt * ((1:pulse_size) - pulse_size/2);
    gaussian_pulse = exp(-(t .* t) / (2 * sigma * sigma));
    
    sin_wave = sin(2*pi*(1/cycle_size * (1:pulse_size)));
    cos_wave = cos(2*pi*(1/cycle_size * (1:pulse_size)));
    
    cos_pulse = single((cos_wave .* gaussian_pulse));
    sin_pulse = single((sin_wave .* gaussian_pulse));
    
%     figure;
%     hold on;
%     title('Phasor pulse')
%     p1 = plot(cos_pulse, 'b');
%     p2 = plot(sin_pulse, 'g');
%     p3 = plot(gaussian_pulse, 'r');
%     legend([p1, p2, p3], ...
%         {'cos kernel','sin kernel','gauss envelope'}, ...
%         'Location', 'Northeast')
%     hold off;
%     
    % Convolution time response with virtual wave kernel 
    [Nt, Nx, Ny] = size(dataCube); 
    P_real = single(zeros(Nt, Nx,Ny));
    P_imag = single(zeros(Nt, Nx,Ny));
    for p = 1 : Nx
        for c = 1 : Ny
            P_real(:,p, c) = conv(dataCube(:,p,c), cos_pulse, 'same');
            P_imag(:,p, c) = conv(dataCube(:,p,c), sin_pulse, 'same');
        end
    end

end

