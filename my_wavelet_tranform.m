function coeff = my_wavelet_tranform(img, wavelet_op)
    [ca, cb, cc, cd] = dwt2(img, wavelet_op);
    coeff = [ca,cb,cc,cd];  
end