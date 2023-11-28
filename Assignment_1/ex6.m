%% EX 6
clearvars; close all; clc

% Statement of the problem
x0 = [1 1]';
tlim = [0 5];
% Analytical solution
B = [-180.5 219.5; 179.5 -220.5];
eig_B = eig(B)*0.1;
x_an = @(t) expm(t*B)*x0;
f = @(t, x) B*x;
% RK4 solution
[t_rk4, y_rk4, ~, ~] = RK4(f, tlim, 0.1, x0);
N = 1000;
A = @(alpha) [0 1; -1 2*cos(alpha)];
% ------------------------- Solution with IEX4 --------------------------
% step
h = 0.1;
t_v = tlim(1):h:tlim(2);
y_IEX4 = [x0, zeros(2, length(t_v)-1)];
y_an = [x0, zeros(2, length(t_v)-1)];

Fs_IEX4 = @(h) (-1/6)*((eye(2)-h*B))^(-1)+4*(((eye(2)-(h/2)*B)^2))^(-1)-...
    (27/2)*(((eye(2)-(h/3)*B)^3))^(-1)+(32/3)*(((eye(2)-(h/4)*B)^4))^(-1); 

for i = 2:length(t_v)
    y_IEX4(:, i) = Fs_IEX4(h)*y_IEX4(:, i-1);
    y_an(:, i) = x_an(t_v(i));
end

%%%  Stability domains
% The same code to generate the stability domain is taken from Exercise 3
alpha_v = linspace(0,pi,N);
FRK4 = @(h,a) eye(2) + h*A(a) + 0.5*h^2*A(a)^2 +...
    1/6*h^3*A(a)^3 + 1/24*h^4*A(a)^4;
[R_RK4,~] = solveStabilityRegion(FRK4, alpha_v, A, 3, 2);
re_RK4 = real(R_RK4);
im_RK4 = imag(R_RK4);
reTot_RK4 = [re_RK4; re_RK4(end:-1:1)];
imTot_RK4 = [im_RK4; -im_RK4(end:-1:1)];
%  ----------------------- IEX4 instability domain ----------------------
hIEX4  = zeros(length(alpha_v), 1);
R_IEX4 = zeros(length(alpha_v), 1);

FIEX4 = @(h, a) (-1/6)*((eye(2)-h*A(a)))^(-1)+4*(((eye(2)-(h/2)*A(a))^2))^(-1)-...
    (27/2)*(((eye(2)-(h/3)*A(a))^3))^(-1)+(32/3)*(((eye(2)-(h/4)*A(a))^4))^(-1); 
% intial guess
hg = 10;
it = 1;
for alpha = alpha_v
    fun = @(h) max(abs(eig(FIEX4(h, alpha)))) - 1;
    hIEX4(it) = fzero(fun, hg);
    hg = hIEX4(it);
    lambda = eig(A(alpha));
    R_IEX4(it) = hIEX4(it)*lambda(1);
    it = it + 1;
end
re_IEX4 = real(R_IEX4);
im_IEX4 = imag(R_IEX4);
reTot_IEX4 = [re_IEX4; re_IEX4(end:-1:1)];
imTot_IEX4 = [im_IEX4; -im_IEX4(end:-1:1)];
%%% -------------------------------- PLOTS -----------------------------
% Stability/Instability domain
fig = figure();
plot(reTot_RK4, imTot_RK4, '-.', 'LineWidth', 1.5); 
hold on; 
grid on;
plot(reTot_IEX4, imTot_IEX4, 'LineWidth', 1.5);
fill(reTot_RK4, imTot_RK4, 'b', 'FaceAlpha', 0.4);
fill(reTot_IEX4, imTot_IEX4, 'r', 'FaceAlpha', 0.4);
plot(eig_B(1), 0, 'kx', 'markerSize', 10);
plot(eig_B(2), 0, 'ko', 'markerSize', 10, 'markerFaceColor', 'g');
xLIM = xlim; 
xlim([xLIM(1)-0.5 xLIM(2)+0.5]);
yLIM = ylim; 
ylim([yLIM(1)-0.5 yLIM(2)+0.5]);
axis equal; 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend({'RK4 Stability margin', 'IEX4 Stability Margin',...
    'RK4 Stability region', 'IEX4 Instability region', ...
    '$h\lambda_1$ of matrix B', '$h\lambda_2$ of matrix B'},...
    'FontSize', 14, 'Location', 'southwest');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
% create zoom to show the eigenvalue
a2 = axes();
a2.Position = [0.15 0.6 0.4 0.25];
box on
hold on; 
axis equal; 
grid on;
fill(reTot_IEX4, imTot_IEX4, 'r', 'FaceAlpha', 0.4);
fill(reTot_RK4, imTot_RK4, 'b', 'FaceAlpha', 0.4);
plot(eig_B(1), 0, 'kx', 'markerSize', 10);
ax = gca;
ax.XLim = [-0.8 0.8];
ax.YLim = [-0.5 0.5];
%save_fig(fig,'ex6_1');

% RK4 solution
fig =  figure();
plot(t_rk4, y_rk4(:, 1), 'LineWidth', 1.5); 
grid on; 
hold on;
plot(t_rk4, y_rk4(:, 2), '--', 'LineWidth', 1.5);
legend({'$x_1$', '$x_2$'}, 'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$x$ [-]', 'FontSize', 18);
title('RK4 - solution', 'FontSize', 17);
%save_fig(fig,'ex6_2');

% IEX4 solution
fig = figure();
plot(t_rk4, y_IEX4(1, :), 'LineWidth', 1.5); 
grid on; 
hold on;
plot(t_rk4, y_IEX4(2, :), '--', 'LineWidth', 1.5);
legend({'$x_1$', '$x_2$'}, 'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$x$ [-]', 'FontSize', 18);
title('IEX4 - solution', 'FontSize', 17);
%save_fig(fig,'ex6_4');

% RK4 local integration error
fig = figure();
semilogy(t_v, abs(y_an(1, :) - y_rk4(:, 1)'), 'LineWidth', 1.5);
grid on; 
hold on;
semilogy(t_v, abs(y_an(2, :) - y_rk4(:, 2)'), '--', 'LineWidth', 1.5);
legend({'Error on first component', 'Error on second component'},...
    'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('Error [-]', 'FontSize', 18);
title('Error between analytical and RK4 method', 'FontSize', 17);
%save_fig(fig,'ex6_3');

% IEX4 local integration error
fig = figure();
semilogy(t_v, abs(y_an(1, :) - y_IEX4(1, :)), 'LineWidth', 1.5); 
grid on; 
hold on;
semilogy(t_v, abs(y_an(2, :) - y_IEX4(2, :)), '-.', 'LineWidth', 1.5);
legend({'Error on first component', 'Error on second component'},...
    'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('Error [-]', 'FontSize', 18);
title('Error between analytical and IEX4 method', 'FontSize', 17);
%save_fig(fig,'ex6_5');
%system('cd ../Report && pdflatex main.tex > NUL');



function [t, Y, ct, fcnE] = RK4(f, tlim, h, x0)
% RK4 - Solve the ODE problem using the second-order Runge-Kutta (RK2) method.
%
%   [t, Y, ct, fcnE] = RK4(f, tlim, h, x0)
%
% INPUT:
%   f      - Function handle [1x1] for evaluating the ordinary differential equation (ODE).
%   tlim   - Double [1x2] specifying the time vector limits [initial, final].
%   h      - Double [1x1] representing the step size.
%   x0     - Double [nx1] representing the initial value(s).
%
% OUTPUT:
%   t      - Double [1xN] representing the time vector.
%   Y      - Double [mxN] representing the solution matrix.
%   ct     - Double [1x1] representing the computational time needed in seconds.
%   fcnE   - Double [1x1] representing the number of function evaluations.
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

N = round((tlim(2) - tlim(1))/h);
t = linspace(tlim(1), tlim(2), N+1);
Y = [x0, zeros(length(x0), N)];

fcnE = 0;

for i = 2:N+1
    K1 = f(t(i-1), Y(:, i-1));
    K2 = f(t(i-1) + h/2, Y(:, i-1) + h*K1/2);
    K3 = f(t(i-1) + h/2, Y(:, i-1) + h*K2/2);
    K4 = f(t(i), Y(:, i-1) + h*K3);
    Y(:, i) = Y(:, i-1) + h/6*(K1 + 2*K2 + 2*K3 + K4);
    
    fcnE = fcnE + 4;
end

Y = Y';
ct = toc(t0);

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
%h0Vec = 3;
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

