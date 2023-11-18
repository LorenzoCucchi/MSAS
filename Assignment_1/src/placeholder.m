function F = FRK2(h, alpha)
% FRK2 - Retrieve the linear operator of the second-order Runge-Kutta (RK2) method.
%
%   F = FRK2(h, alpha)
%
% INPUT:
%   h      - Double [1x1] representing the step size.
%   alpha  - Double [1x1] representing the angle of eigenvalue in the hlambda axis.
%
% OUTPUT:
%   F      - Double [2x2] representing the linear operator.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   FRK2 retrieves the linear operator of the second-order Runge-Kutta (RK2)
%   method. The linear operator is computed based on the given step size 'h'
%   and the angle of the eigenvalue in the hlambda axis 'alpha'.
%
% ----------------------------------------------------------------------

A = [0 1; -1 2*cos(alpha)];
F = eye(2) + h*A + 0.5*h^2*A^2;
end

function F = FRK4(h, alpha)
% FRK4 - Retrieve the linear operator of the fourth-order Runge-Kutta (RK4) method.
%
%   F = FRK4(h, alpha)
%
% INPUT:
%   h      - Double [1x1] representing the step size.
%   alpha  - Double [1x1] representing the angle of eigenvalue in the hlambda axis.
%
% OUTPUT:
%   F      - Double [2x2] representing the linear operator.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   FRK4 retrieves the linear operator of the fourth-order Runge-Kutta (RK4)
%   method. The linear operator is computed based on the given step size 'h'
%   and the angle of the eigenvalue in the hlambda axis 'alpha'.
%
% -------------------------------------------------------------------------

A = [0 1; -1 2*cos(alpha)];
F = eye(2) + h*A + 0.5*h^2*A^2 + 1/6*h^3*A^3 + 1/24*h^4*A^4;

end

hF  = nan(length(alphaVec), 3);
it = 1;
h0Vec = linspace(0, h_max, 3);
%h0Vec = 3;
opt = optimset('TolX', 1e-15);

for alpha = alphaVec
    fun = @(h) max(abs(eig(F(h, alpha)))) - 1;
    
    % Used to account for other zeros of the function
    sol = nan(length(h0Vec), 1);
    for i = 1:length(h0Vec)
        sol(i) = fzero(fun, h0Vec(i), opt);
    end
    S = nan(1, 3);
    s0 = sort(unique(abs(round(sol, 4))));
    S(1:length(s0)) = s0;
    hF(it, 1:length(S)) = S;
    
    it = it + 1;
end

it = 1;
R_F = nan(length(alphaVec), 1);
App = zeros(size(R_F));
hF_sorted = zeros(length(alphaVec), 1);
alphaU    = zeros(length(alphaVec), 1);
for i = 1:length(alphaVec)

    lambda = eig(A(alphaVec(i))); 
    lambda = lambda(1);
    numNaN = sum(isnan(hF(i, :)));
    switch numNaN
        case 1
            hF_sorted(it) = hF(i, 2);
        case 2
            hF_sorted(it) = hF(i, 1);
        case 0
            for j = 2:3
                hF_sorted(it) = hF(i, j);
                alphaU(it) = alphaVec(i);
                R_F(it) = hF_sorted(it) * lambda;
                it = it + 1;
            end
    end

    if numNaN ~= 0
        alphaU(it) = alphaVec(i);
        R_F(it) = hF_sorted(it) * lambda;
        it = it + 1;
    end
    App(i) = atan2(imag(R_F(i)), real(R_F(i))+corr);
end
App(end+1) = 0;
R_F(end+1) = 0;
alphaU(end+1) = pi/2;
[~, ind] = sort(App);
R_F = R_F(ind);
alphaU = alphaU(ind);