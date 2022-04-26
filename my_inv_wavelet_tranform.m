function img = my_inv_wavelet_tranform(x,wavelet_op)
    [ca, cb, cc, cd] = split_waveletCoeff(x);
    img = idwt2(ca,cb,cc,cd,wavelet_op);   
end