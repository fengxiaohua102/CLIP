function Image_FM = FocusMeasure(Image, Method)
%***** this function computes the different focus measure used to
%  detect the sharpest focus for DFF application in Light field
%  the following focus measure are currently implemented
%
%  1: SML: sum of modified laplacian
%  2: IME: image energy
%  3: RDF
switch Method
    case 'SML'
        hx = [-1,2,-1]/4;
        Image_xx = conv2(hx, 1, Image,'same');
        Image_yy = conv2(1, hx, Image,'same');
        Image_FM = abs(Image_xx) + abs(Image_yy);
        Image_FM = imgaussfilt(Image_FM,5);
    case 'RSML'
        hx = [-1,0,2,0,-1]/4;
        Image_xx = conv2(hx, 1, Image,'same');
        Image_yy = conv2(1, hx, Image,'same');
        Image_FM = abs(Image_xx) + abs(Image_yy);
        Image_FM = imgaussfilt(Image_FM,3);
        
    case 'RSML2'
        hx = [-1,-1,0,4,0,-1,-1]/2;
        Image_xx = conv2(hx, 1, Image,'same');
        Image_FM = abs(Image_xx);
        Image_FM = imgaussfilt(Image_FM,3);
    
    case 'RSML3'
        hx = [-1,0, -1,0,-2,0,8,0,-2,0,-1,0,-1]/8;
        Image_xx = conv2(hx, 1, Image,'same');
        Image_FM = abs(Image_xx);
        Image_FM = imgaussfilt(Image_FM,3);
        
    case 'IME'
        h = ones(13,13)./169;
        Image_IME = Image.^2;
        Image_FM = conv2(Image_IME, h,'same');
    
    case 'SPARC'
        fun = @(x) nnz(x(:));
        Image_FM = -nlfilter(Image, [15,15],fun);
        
    case 'RDF'
        h = zeros(15,15);
        x = (1:15)-8;
        y = x;
        [x,y] = meshgrid(x,y);
        inner_ring = ((x.^2+y.^2)<=2^2);
        mask_ring =  ((x.^2+y.^2)>4^2) & (((x.^2+y.^2)<=8^2));
        h(mask_ring) = -2./nnz(mask_ring);
        h(inner_ring) = 2./nnz(inner_ring);
        Image_FM = conv2(Image,h,'same');
        Image_FM = imgaussfilt(Image_FM,5);
        
    otherwise
        h = ones(5,5);
        Image_IME = Image.^2;
        Image_FM = conv2(Image_IME, h,'same');
end
        
end