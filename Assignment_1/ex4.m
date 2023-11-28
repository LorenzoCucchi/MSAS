%% EX 4
clearvars; close all; clc

% Statement of the problem
x0 = [1 1]';
A = @(alpha) [0 1; -1 2*cos(alpha)];
t_int = [0 1];
t_d = diff(t_int);
% Analytical solution
x_an = @(t, alpha) expm(t*A(alpha))*x0;

% Linear operator of the methods
F_RK1 = @(h, alpha) eye(length(x0)) + h*A(alpha);
F_RK2 = @(h, alpha) eye(length(x0)) + h*A(alpha) + (h^2)/2*A(alpha)^2;
F_RK4 = @(h, alpha) eye(length(x0)) + h*A(alpha) + (h^2)/2*A(alpha)^2 ...
                            + ((h^3)/6)*A(alpha)^3 + ((h^4)/24)*A(alpha)^4;

%%% CALCULATIONS ----------------------------------------------------------
% Pre-allocation
alpha_v = linspace(0, pi, 100);
tolVec = [1e-3, 1e-4, 1e-5, 1e-6];

[h_rk1, h_rk2, h_rk4, r_rk1, r_rk2, r_rk4] = deal(cellfun(@(x)...
    zeros(length(alpha_v), 1), cell(length(tolVec), 1), 'UniformOutput', false));
% get the actual number of iterations given h
N = @(h) (t_int(2)-rem(t_d, h))/h;

% Loop to find all the h values that zeroed the functions fun(i)
for i = 1:length(tolVec)
    toll = tolVec(i);
    for j = 1:length(alpha_v)
        a_v = alpha_v(j);
        x_fan = x_an(t_int(2), a_v);
        
        fun1 = @(h) norm(x_fan - F_RK1(rem(t_d, h), a_v)*(F_RK1(h, a_v)^N(h))*x0, 'inf') - toll;
        fun2 = @(h) norm(x_fan - F_RK2(rem(t_d, h), a_v)*(F_RK2(h, a_v)^N(h))*x0, 'inf') - toll;
        fun4 = @(h) norm(x_fan - F_RK4(rem(t_d, h), a_v)*(F_RK4(h, a_v)^N(h))*x0, 'inf') - toll;
        
        h_rk1{i}(j) = fsolve(fun1, 5e-5, optimoptions('fsolve', 'Display', 'none'));
        h_rk2{i}(j) = fzero(fun2, [5e-6, 1]);
        h_rk4{i}(j) = fzero(fun4, [5e-6, 1]);
        
        lambda = eig(A(alpha_v(j))); 
        r_rk1{i}(j) = h_rk1{i}(j)*lambda(1);
        r_rk2{i}(j) = h_rk2{i}(j)*lambda(1);
        r_rk4{i}(j) = h_rk4{i}(j)*lambda(1);
    end
    
end

[feval1, feval2, feval4] = deal(zeros(length(tolVec), 1));
A_pi = A(pi);
% Obtain function evaluations number
for i = 1:length(tolVec)
    toll = tolVec(i);
    [~, ~, ~, feval1(i)] = RK1(@(t, x) A_pi*x, t_int, h_rk1{i}(end), x0);
    [~, ~, ~, feval2(i)] = RK2(@(t, x) A_pi*x, t_int, h_rk2{i}(end), x0);
    [~, ~, ~, feval4(i)] = RK4(@(t, x) A_pi*x, t_int, h_rk4{i}(end), x0);
end

%%% -------------------------------- PLOTS --------------------------------
% RK1 solutions
fig = figure();
lineSpecObj = {'-.', '--', ':'};
for i = 1:length(tolVec)
    re = real(r_rk1{i});
    im = imag(r_rk1{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    if i == 1
        fp(i) = plot(reL, imL, 'LineWidth', 1.5); hold on; grid on;
    else
        fp(i) = plot(reL, imL, lineSpecObj{i-1}, 'LineWidth', 1.5);
    end
end
axis equal; 
ax = gca;
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend(fp, {'Tol = $10^{-3}$', 'Tol = $10^{-4}$', 'Tol = $10^{-5}$', 'Tol = $10^{-6}$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
title('RK1', 'FontSize', 17);
a2 = axes();
a2.Position = [0.18 0.17 0.25 0.25];
box on
hold on; axis equal; grid on;
for i = 1:length(tolVec)
    re = real(r_rk1{i});
    im = imag(r_rk1{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    if i == 1
        fp(i) = plot(reL, imL, 'LineWidth', 1.5); hold on; grid on;
    else
        fp(i) = plot(reL, imL, lineSpecObj{i-1}, 'LineWidth', 1.5);
    end
end
xlim([-2e-5 3e-5]); 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
%save_fig(fig,'ex4_1');


% RK2 solutions
fig = figure();
lineSpecObj = {'-.', '--', ':'};
for i = 1:length(tolVec)
    re = real(r_rk2{i});
    im = imag(r_rk2{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    if i == 1
        fp(i) = plot(reL, imL, 'LineWidth', 1.5); hold on; grid on;
    else
        fp(i) = plot(reL, imL, lineSpecObj{i-1}, 'LineWidth', 1.5);
    end
end
axis equal; 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend(fp, {'Tol = $10^{-3}$', 'Tol = $10^{-4}$', 'Tol = $10^{-5}$', 'Tol = $10^{-6}$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
title('RK2', 'FontSize', 17);
%save_fig(fig,'ex4_2');


% RK4 solutions
fig = figure();
lineSpecObj = {'-.', '--', ':'};
for i = 1:length(tolVec)
    re = real(r_rk4{i});
    im = imag(r_rk4{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    if i == 1
        fp(i) = plot(reL, imL, 'LineWidth', 1.5); hold on; grid on;
    else
        fp(i) = plot(reL, imL, lineSpecObj{i-1}, 'LineWidth', 1.5);
    end
end
axis equal; 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend(fp, {'Tol = $10^{-3}$', 'Tol = $10^{-4}$', 'Tol = $10^{-5}$', 'Tol = $10^{-6}$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
title('RK4', 'FontSize', 17);
%save_fig(fig,'ex4_3');

% Function evaluations vs tolerance plot
fig = figure();
loglog(tolVec, feval1, '-s', 'LineWidth', 1.5); hold on;
loglog(tolVec, feval2, '-^', 'LineWidth', 1.5)
loglog(tolVec, feval4, '-o', 'LineWidth', 1.5); grid on;
title('f evaluations vs Tolerance', 'FontSize', 17);
xlabel('tol [-]', 'FontSize', 18); 
ylabel('f evaluations [-]', 'FontSize', 18);
legend({'RK1', 'RK2', 'RK4'}, 'FontSize', 14, 'Location', 'best');
%save_fig(fig,'ex4_4');
%system('cd ../Report && pdflatex main.tex > NUL');



function [t, Y, ct, feval] = RK1(f, t_int, h, x0)
% RK1 - Solve the ODE problem using the first-order Runge-Kutta (RK1) method.
%
%   [t, Y, ct, feval] = RK1(f, t_int, h, x0)
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
%   feval   - Double [1x1] representing the number of function evaluations.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   RK1 solves the ordinary differential equation (ODE) problem described by
%   the function f using the first-order Runge-Kutta (RK1) approximation. The
%   function iteratively updates the solution based on the RK1 method until
%   the specified final time is reached.
%
% -------------------------------------------------------------------------

t0 = tic;

N = round((t_int(2) - t_int(1))/h);
t = linspace(t_int(1), t_int(2), N+1);
Y = [x0, zeros(length(x0), N)];

feval = 0;

for i = 2:N+1
    K1 = f(t(i-1), Y(:, i-1));
    Y(:, i) = Y(:, i-1) + h*K1;
    
    feval = feval + 1;
end

Y = Y';
ct = toc(t0);

end

function [t, Y, ct, feval] = RK2(ode, t_int, h, x0)
% RK2 - Solve the ODE problem using the second-order Runge-Kutta (RK2) method.
%
%   [t, Y, ct, feval] = RK2(f, t_int, h, x0)
%
% INPUT:
%   ode      - Function handle [1x1] for evaluating the ordinary differential equation (ODE).
%   t_int   - Double [1x2] specifying the time vector limits [initial, final].
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
    k2 = ode(t(i-1) + h, Y(:, i-1) + k1*h);
    Y(:, i) = Y(:, i-1) + h*(1/2*k1 + 1/2*k2);
    feval = feval + 2;
end
Y = Y';
ct = toc(t0);

end

function [t, Y, ct, feval] = RK4(ode, t_int, h, x0)
% RK4 - Solve the ODE problem using the second-order Runge-Kutta (RK2) method.
%
%   [t, Y, ct, feval] = RK4(f, t_int, h, x0)
%
% INPUT:
%   ode      - Function handle [1x1] for evaluating the ordinary differential equation (ODE).
%   t_int   - Double [1x2] specifying the time vector limits [initial, final].
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
