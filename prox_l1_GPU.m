function sol = prox_l1_GPU(x, gamma)
    temp = x;
    
    % the soft_threshold step
    sol = soft_threshold(temp, gamma);    
    
    % norm_l1 = gamma*sum(abs(sol(:)));
    
    function sz = soft_threshold(z,T)
        if T>0
            size_z = size(z);
            z = z(:);
            T = T(:);

            % 1: This soft thresholding function only supports real signal
            sz = sign(z).*max(abs(z)-T, 0);

            % 2: This soft thresholding function supports complex numbers
%             sz = max(abs(z)-T,0)./(max(abs(z)-T,0)+T).*z;
            % Handle the size
            sz = reshape(sz,size_z);
        else
            sz = z;
        end
    end
end