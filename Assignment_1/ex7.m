%% EX 7
clearvars; close all; clc

% Statement of the problem
x0 = [1 1]';
t_int = [0 3];
h = 0.1;

f = @(t, x) [-5/2*(1 + 8*sin(t))*x(1);... 
              (1 - x(1))*x(2) + x(1)];

% Numerical solution
[t_AB3, Y_AB3]   =  AB3(f, t_int, h, x0);
[t_AM3, Y_AM3]   =  AM3(f, t_int, h, x0);
[t_ABM3, Y_ABM3] = ABM3(f, t_int, h, x0);
[t_BDF3, Y_BDF3] = BDF3(f, t_int, h, x0);

%%% STABILITY DOMAIN --------------------------------------------------------
A = @(alpha) [0 1; -1 2*cos(alpha)];
n = size(x0, 1);

% AM3
F_AM3 = @(h, alpha) max(abs(eig([zeros(n, n) eye(n); ...
         -(eye(n) - 5/12*h*A(alpha))\eye(n)*1/12*h*A(alpha),...
         (eye(n) - 5/12*h*A(alpha))\eye(n)*(eye(n) + 2/3*h*A(alpha))])));
% For the stability region code from Ex3 and Ex6 is reused
% AM3 stability region
N = 500;
alpha_v = linspace(pi, 0, N);
R_AM3 = zeros(N, 1);
hGuess = 5;
it = 1;
for alpha = alpha_v
    fun = @(h) max(abs(eig(F_AM3(h, alpha)))) - 1;
    h = fzero(fun, hGuess);
    hGuess = h;
    lambda = eig(A(alpha)); 
    R_AM3(it) = h*lambda(1);
    it = it + 1;
end
re_AM3 = real(R_AM3);
im_AM3 = imag(R_AM3);
reTot_AM3 = [re_AM3; re_AM3(end:-1:1)];
imTot_AM3 = [im_AM3; -im_AM3(end:-1:1)];

% AB3, ABM3 and BDF3 stability regions
% The alpha_v is not evenly spaced in order to better refine plots in 
% certain areas
alpha_v = [linspace(0, 80*pi/180, 1/4*N),...
             linspace(80*pi/180, pi/2, 1/2*N),...
                linspace(pi/2, pi, 1/4*N)];

% AB3
F_AB3 = @(h, alpha) [zeros(n, n) eye(n) zeros(n, n); ...
         zeros(n, n) zeros(n, n) eye(n); ...
         5/12*A(alpha)*h -4/3*A(alpha)*h (eye(n) + 23/12*A(alpha)*h)];
hm_AB3 = 1;
[R_AB3,~] = solveStabilityRegion(F_AB3, alpha_v, A, 1, 0.4);
re_AB3 = real(R_AB3);
im_AB3 = imag(R_AB3);
reTot_AB3 = [re_AB3; re_AB3(end:-1:1)];
imTot_AB3 = [im_AB3; -im_AB3(end:-1:1)];

% ABM3
F_ABM3 = @(h, alpha) max(abs(eig([zeros(n, n) eye(n) zeros(n, n); ...
         zeros(n, n) zeros(n, n) eye(n); ...
         25/144*(A(alpha)*h)^2, -(1/12*A(alpha)*h + 5/9*(A(alpha)*h)^2),...
         (eye(n) + 13/12*A(alpha)*h + 115/144*(A(alpha)*h)^2)])));
hm_ABM3 = 2;
[R_ABM3,~] = solveStabilityRegion(F_ABM3, alpha_v, A, 2, 1.5);
re_ABM3 = real(R_ABM3);
im_ABM3 = imag(R_ABM3);
reTot_ABM3 = [re_ABM3; re_ABM3(end:-1:1)];
imTot_ABM3 = [im_ABM3; -im_ABM3(end:-1:1)];

% BDF3
F_BDF3 = @(h, alpha) max(abs(eig([zeros(n, n) eye(n) zeros(n, n); ...
         zeros(n, n) zeros(n, n) eye(n); ...
         eye(n)/(eye(n) - 6/11*A(alpha)*h)*2/11, - (eye(n) -...
         6/11*A(alpha)*h)\eye(n)*9/11, ((eye(n) - 6/11*A(alpha)*h)\eye(n))*18/11])));
hm_BDF3 = 6;
[R_BDF3,~] = solveStabilityRegion(F_BDF3, alpha_v, A, 6, -5);
re_BDF3 = real(R_BDF3);
im_BDF3 = imag(R_BDF3);
reTot_BDF3 = [re_BDF3; re_BDF3(end:-1:1)];
imTot_BDF3 = [im_BDF3; -im_BDF3(end:-1:1)];
%%% -------------------------------- PLOTS ------------------------------
% AB3 solution
fig = figure('Name', 'EX 7 - AB3', 'NumberTitle', 'off');
plot(t_AB3, Y_AB3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_AB3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('AB3', 'FontSize', 17);
legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
%save_fig(fig,'ex7_1');

% AM3 solution
fig = figure('Name', 'EX 7 - AM3', 'NumberTitle', 'off');
plot(t_AB3, Y_AM3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_AM3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('AM3', 'FontSize', 17);
legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
%save_fig(fig,'ex7_2');

% ABM3 solution
fig = figure('Name', 'EX 7 - ABM3', 'NumberTitle', 'off');
plot(t_AB3, Y_ABM3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_ABM3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('ABM3', 'FontSize', 17);
legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
%save_fig(fig,'ex7_3');

% BDF3 solution
fig = figure('Name', 'EX 7 - BDF3', 'NumberTitle', 'off');
plot(t_AB3, Y_BDF3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_BDF3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('BDF3', 'FontSize', 17);
legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
%save_fig(fig,'ex7_4');

% Real system eigenvalues
M = @(t) [-5/2*(1 + 8*sin(t)) 0; 1 1];
lambda = zeros(2, length(t_AB3));
for i = 1:length(t_AB3)
    lambda(:, i) = eig(M(t_AB3(i)))*0.1;
end
fig = figure('Name', 'EX 7 - Real system eigenvalues', 'NumberTitle', 'off');
plot(t_AB3, lambda(2, :), 'LineWidth', 1.5); grid on; hold on;
plot(t_AB3, lambda(1, :), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$h\lambda_i$ [-]', 'FontSize', 18);
title('$h\lambda_i$ of the linearized system', 'FontSize', 17);
legend('$\lambda_x$', '$\lambda_y$', 'FontSize', 14, 'Location','best');
% save_fig(fig,'ex7_5');

% Stability/Instability domains
fig = figure('Name', 'EX 7 - Stability region', 'NumberTitle', 'off');
plot(reTot_AB3, imTot_AB3, 'LineWIdth', 1.5); 
hold on; 
grid on;
plot(reTot_AM3, imTot_AM3, '-.', 'LineWidth', 1.5);
plot(reTot_ABM3, imTot_ABM3, ':', 'LineWidth', 1.5);
plot(reTot_BDF3, imTot_BDF3);
xLIM = xlim; xlim([xLIM(1)-0.5 xLIM(2)+0.5]);
yLIM = ylim; ylim([yLIM(1)-0.5 yLIM(2)+0.5]);
axis equal; 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend({'AB3 Stability margin', 'AM3 Stability margin',...
    'ABM3 Stability region', 'BDF3 Instability region'},...
    'FontSize', 12, 'Location', 'southeast');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
% save_fig(fig,'ex7_6');

function [t, Y, ct, feval] = RK4(ode, t_int, h, x0)
% RK4 - Solve the ODE problem using the second-order Runge-Kutta (RK2) method.
%
%   [t, Y, ct, feval] = RK4(f, t_int, h, x0)
%
% INPUT:
%   ode    - Function handle [1x1] for evaluating the ordinary differential equation (ODE).
%   t_int  - Double [1x2] specifying the time vector limits [initial, final].
%   h      - Double [1x1] representing the step size.
%   x0     - Double [nx1] representing the initial value(s).
%
% OUTPUT:
%   t      - Double [1xN] representing the time vector.
%   Y      - Double [mxN] representing the solution matrix.
%   ct     - Double [1x1] representing the computational time needed in seconds.
%   feval   - Double [1x1] representing the number of function evaluations.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   RK2 solves the ordinary differential equation (ODE) problem described by
%   the function f using the second-order Runge-Kutta (RK2) approximation. The
%   function iteratively updates the solution based on the RK2 method until
%   the specified final time is reached.
%
% -------------------------------------------------------------------------

t0 = tic;
N = round((t_int(2) - t_int(1))/h);
t = linspace(t_int(1), t_int(2), N+1);
Y = [x0, zeros(length(x0), N)];
feval = 0;
for i = 2:N+1
    k1 = ode(t(i-1), Y(:, i-1));
    k2 = ode(t(i-1) + h/2, Y(:, i-1) + h*k1/2);
    k3 = ode(t(i-1) + h/2, Y(:, i-1) + h*k2/2);
    k4 = ode(t(i), Y(:, i-1) + h*k3);
    Y(:, i) = Y(:, i-1) + h/6*(k1 + 2*k2 + 2*k3 + k4);
    feval = feval + 4;
end

Y = Y';
ct = toc(t0);

end

%%% ---------------------------- EX-7 -----------------------------------

function [t, Y] = AB3(f, t_int, h, x0)
% AB3 - Solve the ODE problem using the third-order Adams-Bashforth (AB3) method.
%
%   [t, Y, ct] = AB3(f, t_int, h, x0)
%
% INPUT:
%   f      - Function handle [1x1] for evaluating the ordinary differential equation (ODE).
%   t_int   - Double [1x2] specifying the time vector limits [initial, final].
%   h      - Double [1x1] representing the step size.
%   x0     - Double [nx1] representing the initial value(s).
%
% OUTPUT:
%   t      - Double [1xN] representing the time vector.
%   Y      - Double [mxN] representing the solution matrix.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   AB3 solves the ordinary differential equation (ODE) problem described by
%   the function f using the third-order Adams-Bashforth (AB3) approximation.
%   The function iteratively updates the solution until the specified final
%   time is reached.
%
% -------------------------------------------------------------------------

N = (t_int(2) - t_int(1))/h;
t = linspace(t_int(1), t_int(2), N+1);
% First 2 steps are obtained with RK4 method
[~, Y, ~, ~] = RK4(f, [0 2*h], h, x0);

for i = 4:N+1
    Y(i, :) = Y(i-1, :)' + h*(23/12*f(t(i-1), Y(i-1, :)')...
               - 16/12*f(t(i-2), Y(i-2, :)') + 5/12*f(t(i-3), Y(i-3, :)'));
end

end


function [t, y] = AM3(f, t_int, h, x0)
% AM3 - Solve the ODE problem using the third-order Adams-Moulton (AM3) method.
%
%   [t, Y, ct] = AM3(f, t_int, h, x0)
%
% INPUT:
%   f      - Function handle [1x1] for evaluating the ordinary differential equation (ODE).
%   t_int   - Double [1x2] specifying the time vector limits [initial, final].
%   h      - Double [1x1] representing the step size.
%   x0     - Double [nx1] representing the initial value(s).
%
% OUTPUT:
%   t      - Double [1xN] representing the time vector.
%   Y      - Double [mxN] representing the solution matrix.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   AM3 solves the ordinary differential equation (ODE) problem described by
%   the function f using the third-order Adams-Moulton (AM3) approximation.
%   The function iteratively updates the solution until the specified final
%   time is reached.
%
% -------------------------------------------------------------------------

toll = 1e-8;
N = (t_int(2) - t_int(1))/h;
t = linspace(t_int(1), t_int(2), N+1);
% First 2 steps are obtained with RK4 method
[~, y, ~, ~] = RK4(f, [0 2*h], h, x0);

% Anlytical inverse Jacobian for Newton's method
invJ = @(t, y) [24/(25*h + 200*h*sin(t) + 24), 0; ...
                    -(120*h*(y(2) - 1))/((5*h*y(1) - 5*h + 12)...
                    *(25*h + 200*h*sin(t) + 24)), 12/(5*h*y(1) - 5*h + 12)];

for i = 3:N+1
    err = toll + 1;
    x = y(i-1, :)';
    fun = @(x) x - (y(i - 1, :)' + h * (5 / 12 * [-5 / 2 * (1 + 8 * sin(t(i))) * x(1); ...
                 (1 - x(1)) * x(2) + x(1)] + 8 / 12 * [-5 / 2 * (1 + 8 * sin(t(i) - h)) * y(i - 1, 1); ...
                 (1 - y(i - 1, 1)) * y(i - 1, 2) + y(i - 1, 1)] - ...
                 1 / 12 * [-5 / 2 * (1 + 8 * sin(t(i) - (2 * h))) * y(i - 2, 1); ...
                 (1 - y(i - 2, 1)) * y(i - 2, 2) + y(i - 2, 1)]));
    while any(err > toll)
        fx = fun(x);
        x = x - invJ(t(i), x)*fx;
        err = abs(fx);
    end
    y(i, :) = x;
end

end


function [t, y] = ABM3(f, t_int, h, x0)
% ABM3 - Solve the ODE problem using the third-order Adams-Bashforth-Moulton (ABM3) method.
%
%   [t, Y, ct] = ABM3(f, t_int, h, x0)
%
% INPUT:
%   f      - Function handle [1x1] for evaluating the ordinary differential equation (ODE).
%   t_int   - Double [1x2] specifying the time vector limits [initial, final].
%   h      - Double [1x1] representing the step size.
%   x0     - Double [nx1] representing the initial value(s).
%
% OUTPUT:
%   t      - Double [1xN] representing the time vector.
%   Y      - Double [mxN] representing the solution matrix.
%   ct     - Double [1x1] representing the computational time needed in seconds.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   ABM3 solves the ordinary differential equation (ODE) problem described by
%   the function f using the third-order Adams-Bashforth-Moulton (ABM3)
%   approximation. The function iteratively updates the solution until the
%   specified final time is reached.
%
% -------------------------------------------------------------------------

N = (t_int(2) - t_int(1))/h;
t = linspace(t_int(1), t_int(2), N+1);
k1 = h / 12 * [23; -16; 5];
k2 = h / 12 * [5/2; 8; -1];

% First 2 steps are obtained with RK4 method
[~, y, ~, ~] = RK4(f, [0 2*h], h, x0);

for i = 4:N+1
     xp = y(i-1, :)' + k1(1) * f(t(i-1), y(i-1, :)') + ...
                     k1(2) * f(t(i-2), y(i-2, :)') + ...
                     k1(3) * f(t(i-3), y(i-3, :)');
     y(i, :) = y(i-1, :)' + k2(1) * f(t(i), xp) + ...
                          k2(2) * f(t(i-1), y(i-1, :)') + ...
                          k2(3) * f(t(i-2), y(i-2, :)');
end

end


function [t, y] = BDF3(f, t_int, h, x0)
% BDF3 - Solve the ODE problem using the third-order Backward Differentiation Formula (BDF3) method.
%
%   [t, Y, ct] = BDF3(f, t_int, h, x0)
%
% INPUT:
%   f      - Function handle [1x1] for evaluating the ordinary differential equation (ODE).
%   t_int   - Double [1x2] specifying the time vector limits [initial, final].
%   h      - Double [1x1] representing the step size.
%   x0     - Double [nx1] representing the initial value(s).
%
% OUTPUT:
%   t      - Double [1xN] representing the time vector.
%   Y      - Double [mxN] representing the solution matrix.
%   ct     - Double [1x1] representing the computational time needed in seconds.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   BDF3 solves the ordinary differential equation (ODE) problem described by
%   the function f using the third-order Backward Differentiation Formula (BDF3)
%   approximation. The function iteratively updates the solution until the
%   specified final time is reached.
%
% -------------------------------------------------------------------------

toll = 1e-8;
N = (t_int(2) - t_int(1))/h;
t = linspace(t_int(1), t_int(2), N+1);
% First 2 steps are obtained with RK4 method
[~, y, ~, ~] = RK4(f, [0 2*h], h, x0);

% Analytical inverse Jacobian
invJ = @(t, y) [11 / (15 * h + 120 * h * sin(t) + 11), 0; ...
                -(66 * h * (y(2) - 1)) / ((6 * h * y(1) - 6 * h + 11)...
                * (15 * h + 120 * h * sin(t) + 11)),...
                11 / (6 * h * y(1) - 6 * h + 11)];

for i = 4:N+1
    err = toll + 1;
    x = y(i-1, :)';
    fun = @(x) x - (6/11 * h * [-5/2 * (1 + 8 * sin(t(i))) * x(1);...
                    (1 - x(1)) * x(2) + x(1)] + 18/11 * y(i - 1, :)'...
                        - 9/11 * y(i - 2, :)' + 2/11 * y(i - 3, :)');

    while any(err > toll)
        fx = fun(x);
        x = x - invJ(t(i), x) * fx;
        err = abs(fx);
    end
    y(i, :) = x;
end

end

function [R_F,alphaU] = solveStabilityRegion(F, alpha_v, A, h_max, corr)
% solveStabilityRegion - Compute and analyze the stability region of a given
% linear operator for various parameter values.
%
% PROTOTYPE
%   [R_F, alphaU] = solveStabilityRegion(F, alpha_v, A, h_max, corr)
%
% INPUT:
%   F         - Function handle [1x1] for the linear operator as a function
%               of step size and parameter alpha.
%   alpha_v  - Double [1xM] array specifying the parameter values.
%   A         - Function handle [1x1] for the matrix A as a function of alpha.
%   h_max     - Double [1x1] representing the maximum step size to explore.
%   corr      - Double [1x1] representing the correction term for stability
%               region computation.
%
% OUTPUT:
%   R_F       - Double [1x(M+1)] array representing the computed radii of the
%               stability region for each alpha and one additional point at
%               alpha = pi/2.
%   alphaU    - Double [1x(M+1)] array representing the corresponding alpha
%               values for each point in R_F.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   solveStabilityRegion computes the stability region for a given linear
%   operator F with varying parameters alpha. The function explores the
%   stability region by finding the maximum absolute eigenvalue for different
%   step sizes and parameter values. The output R_F represents the radii of
%   the stability region, and alphaU represents the corresponding parameter
%   values. The additional point at alpha = pi/2 is included for visualization
%   purposes. 
%
% -------------------------------------------------------------------------


hF  = nan(length(alpha_v), 3);
it = 1;
h0Vec = linspace(0, h_max, 3);
opt = optimset('TolX', 1e-15);

for alpha = alpha_v
    fun = @(h) max(abs(eig(F(h, alpha)))) - 1;
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
R_F = nan(length(alpha_v), 1);
hF_sorted = zeros(length(alpha_v), 1);
alphaU    = zeros(length(alpha_v), 1);
for i = 1:length(alpha_v)

    lambda = eig(A(alpha_v(i))); 
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
                alphaU(it) = alpha_v(i);
                R_F(it) = hF_sorted(it) * lambda;
                it = it + 1;
            end
    end
    
    if numNaN ~= 0
        alphaU(it) = alpha_v(i);
        R_F(it) = hF_sorted(it) * lambda;
        it = it + 1;
    end
    
end
App = atan2(imag(R_F), real(R_F)+corr);
App(end+1) = 0;
R_F(end+1) = 0;
alphaU(end+1) = pi/2;
[~, ind] = sort(App);
R_F = R_F(ind);
alphaU = alphaU(ind);

end