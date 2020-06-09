function X = expmTaylor(A)
% Direct taylor series expansion of the matrix exponential.
% Inputs:
% --------
% A: [m x m] double
%
% Outputs:
% --------
% X: [m x m] double
%       matrix exponential of A

X_old = 1000*eye(size(A));
X = zeros(size(A));
k = 0;

while norm(X - X_old, 'fro') > 1e-15
    X_old = X;
    X = X + A^k/factorial(k);
    k = k + 1;
end 