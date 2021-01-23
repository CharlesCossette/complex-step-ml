function [jac, f_bar] = complexStepLie(f, X_bar, n, wedgeOp, direction)
% Calculates the Jacobian of f:G -> R^m with respect to X \in G, evaluated 
% at X_bar.
% 
% Inputs:
% --------
% f: function handle
%       function handle f:G -> R^m to differentiate
% X_bar: [q x q] double
%       matrix Lie group element, derivative evaluation point.
% n: int
%       dimension (degrees of freedom) of X. i.e. dimension of
%       unconstrained parameterization of X
% wedgeOp: function handle
%       to go from R^n to Lie algebra, i.e. wedgeOp: R^n -> g
% direction: string
%       either 'right' or 'left' to choose the right or left derivatives
%       respectively
%
% Outputs:
% --------
% jac: [m x n] double
%       jacobian matrix
% f_bar: [m x 1] double
%       (optional) function value at derivative evaluation point
x = 5
h = 1e-18;
f_bar = f(X_bar);
m = length(f_bar);

jac = zeros(m,n);

for lv1 = 1:n
    e = zeros(n,1);
    e(lv1) = h*1i;
    if strcmp(direction,'right')
        jac(:,lv1) = imag(f(X_bar*expmTaylor(wedgeOp(e))))/h;
    elseif strcmp(direction, 'left')
        jac(:,lv1) = imag(f(expmTaylor(wedgeOp(e))*X_bar))/h;
    else
        error('Not a valid perturbation direction');
    end
end

end