% Normalize a vector 
function output = norm_fro(x)

x=double(x); 
% x = x-min(x(:));
% output=x./norm(x,'fro'); 
output=x./mean(x(:)); 