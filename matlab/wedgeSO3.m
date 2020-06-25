function output = wedgeSO3(x)
% Generates a skew-symmetric cross-product matrix such that 
% U X V = wedgeSO3(U)*V
%
% Inputs:
% --------
% x: [3 x 1] double
%       Vector in R^3 (such as rotation vector)
%
% Outputs:
% --------
% output: [3 x 3] double
%       skew symmatric cross-product matrix

if numel(x) ~= 3
    error('The input must be a vector of length 3.')
end

x1 = x(1);
x2 = x(2);
x3 = x(3);

output = [  0     -x3     x2;
           x3       0    -x1;
          -x2      x1      0];
end
