%% Test 1 - Complex Step Jacobian SO(3)
% %%% Test function
% DCM where we'll evaluate the derivative
phi_hat = [0;0.5;0.2];
C_bar = expm(wedgeSO3(phi_hat));
v = [3;4;5];
y = [1;2;3];
f = @(C) v.'*C*y;

% Analytical gradient
gradA = -v.'*wedgeSO3(C_bar*y)

% Complex step gradient
gradCS = complexStepLie(f,C_bar,3,@wedgeSO3,'left')

% Machine precision baby
assert(norm(gradA - gradCS) < 1e-14)


%% Test 2 - Complex Step Jacobian SO(3)
% %%% Test function
% DCM where we'll evaluate the derivative
phi_hat = [0;0.5;0.2];
C_bar = expm(wedgeSO3(phi_hat));
v = [3;4;5];
y = [0;0;1];
f = @(C) atan(y.'*C*v);

% Analytical gradient
gradA = -(1+(y.'*C_bar*v)^2)^(-1)*(y.'*C_bar*wedgeSO3(v))

% Complex step gradient
gradCS = complexStepLie(f,C_bar,3,@wedgeSO3,'right')

% Machine precision baby
assert(norm(gradA - gradCS) < 1e-15)