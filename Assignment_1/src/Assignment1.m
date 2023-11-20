% Modeling and Simulation of Aerospace System (2023/2024)
% Assignment # 1
% Author:          Lorenzo Cucchi
% Person code:     10650070
% Student ID:      221732
%
% Comments: every exercise is divide in two parts, the first to solve the
%           exercise questions and the second to plot and print data. The
%           first section contains the graphics setup that needs to be
%           executed when the file is opened.
           


%% Graphics Setup
% Set parameters for desired style of plots
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultLineLineWidth', 1.5);
set(gca, 'box', 'on')

%% Ex 1
clearvars; close all; clc

% Function to evaluate
func = @(x) [x(2)^2 - x(1) - 2; -x(1)^2 + x(2) + 10];
% Inverse Jacobian of the function
invJ = @(x) (1/(4*x(1)*x(2)-1))*[1, -2*x(2); 2*x(1), -1];
% Tolerance
toll = 1e-8;  

% First zero
x1 = [1 -4]';
% Analytical method, netwon finite forward, newton finite centered
[sol1a, n1a] = newtAn(func, x1, invJ, toll);
[sol1f, n1f] = newtFF(func, x1, toll);
[sol1c, n1c] = newtFC(func, x1, toll);

% Second zero
x2 = [6 5]';
% Analytical method, netwon finite forward, newton finite centered
[sol2a, n2a] = newtAn(func, x2, invJ, toll);
[sol2f, n2f] = newtFF(func, x2, toll);
[sol2c, n2c] = newtFC(func, x2, toll);

%%% ----------------------------- PRINT DATA ------------------------------
err_1f = norm(sol1a-sol1f);
err_2f = norm(sol2a-sol2f);
err_1c = norm(sol1a-sol1c);
err_2c = norm(sol2a-sol2c);

fprintf('EXERCISE 1\n\n');
fprintf('First zero: \n');
fprintf('newtAn:   [%2.15f, %2.15f]  \n', sol1a);
fprintf('newtFF:   [%2.15f, %2.15f]  \n', sol1f);
fprintf('newtFC:   [%2.15f, %2.15f]  \n\n', sol1c);
fprintf('Second zero: \n');
fprintf('newtAn:   [%2.15f, %2.15f]  \n', sol2a);
fprintf('newtFF:   [%2.15f, %2.15f]  \n', sol2f);
fprintf('newtFC:   [%2.15f, %2.15f]  \n\n', sol2c);
fprintf('Iterations: \n');
fprintf('newtAn:   [%d, %d]  \n', n1a, n2a);
fprintf('newtFF:   [%d, %d]  \n', n1f, n2f);
fprintf('newtFC:   [%d, %d]  \n\n', n1c, n2c);
fprintf('Errors: \n');
fprintf('newtAn:   [-, -]  \n');
fprintf('newtFF:   [%2.15e, %2.15e]  \n', err_1f, err_2f);
fprintf('newtFC:   [%2.15e, %2.15e]  \n', err_1c, err_2c);


%%% -------------------------------- PLOTS --------------------------------

f1 = @(x1, x2) x2.^2 - x1 - 2;
f2 = @(x1, x2) -x1.^2 + x2 + 10;

% Surface plot of the function f1
fig = figure();
[X, Y] = meshgrid(linspace(-5, 5, 500), linspace(-4, 4, 500));
colormap parula
fp(1) = surf(X, Y, f1(X, Y), 'FaceAlpha', 0.7, 'EdgeColor', 'none'); 
grid on; 
hold on;
[x_f, hh] = contour(X, Y, f1(X, Y), [0 0]); 
delete(hh);
fp(2) = plot3(x_f(1, 2:end), x_f(2, 2:end), zeros(size(x_f(1, 2:end))), '--', 'LineWidth', 3, 'color', 'r');
surf([-5 5; -5 5], [-4 -4; 4 4], zeros(2, 2), 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'k');
ax = gca; 
ax.FontSize = 15;
xlabel('$x_1$ $[-]$', 'FontSize', 20); 
ylabel('$x_2$ $[-]$', 'FontSize', 20);
zlabel('$f_x(\mbox{\boldmath$x$})$ $[-]$', 'FontSize', 20);
legend(fp, {'$f_x$', '$f_x = 0$'}, 'FontSize', 18, 'Location', 'best')
save_fig(fig,'ex1_1');

% Surface plot of the function f2
fig = figure();
[X, Y] = meshgrid(linspace(-5, 5, 500), linspace(-11, 8, 500));
colormap parula
fp(1) = surf(X, Y, f2(X, Y), 'FaceAlpha', 0.7, 'EdgeColor', 'none'); 
grid on; 
hold on;
[x_f, hh] = contour(X, Y, f2(X, Y), [0 0]); delete(hh);
fp(2) = plot3(x_f(1, 2:end), x_f(2, 2:end),zeros(size(x_f(1, 2:end))), '--', 'LineWidth', 3, 'color', 'r');
surf([-5 5;-5 5], [-11 -11;8 8], zeros(2, 2), 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'k');
ax = gca; 
ax.FontSize = 15;
xlabel('$x_1$ $[-]$', 'FontSize', 20); 
ylabel('$x_2$ $[-]$', 'FontSize', 20);
zlabel('$f_y(\mbox{\boldmath $x$})$ $[-]$', 'FontSize', 20);
legend(fp, {'$f_y$', '$f_y = 0$'}, 'FontSize', 18, 'Location', 'best')
save_fig(fig,'ex1_2');


% Plot of the solution
fig = figure();
[X, Y] = meshgrid(linspace(-5, 7, 500), linspace(-8, 8, 500));
contour(X, Y, f1(X, Y), [0 0], 'color', [0 0.45 0.74], 'LineWidth', 1.5); 
grid on; 
hold on;
contour(X, Y, f2(X, Y), [0 0], '--', 'color', [0.85 0.33 0.10], 'LineWidth', 1.5);
ax = gca; 
ax.FontSize = 15;
plot(sol1a(1), sol1a(2), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'y');
text(sol1a(1)+0.2, sol1a(2)+0.5, '$\boldmath{z_1}$', 'FontSize', 17);
plot(sol2a(1), sol2a(2), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'y');
text(sol2a(1)+0.2, sol2a(2)+0.8, '$\boldmath{z_2}$', 'FontSize', 17);
xlabel('$x_1$ $[-]$', 'FontSize', 20); 
ylabel('$x_2$ $[-]$', 'FontSize', 20);
legend({'$f_x = 0$', '$f_y = 0$', 'zeros of $\mbox{\boldmath $f$}(\mbox{\boldmath $x$})$'}, 'FontSize', 15, 'Location', 'best')
save_fig(fig,'ex1_3');
system('cd ../Report && pdflatex main.tex > NUL');


%% EX 2
clearvars; close all; clc

% Statement of the problem
x0   = 1; 
t = [0 2];
% IVP
f    = @(t, x) x - 2*t^2 + 2;
% Analytical solution
u_an  = @(t) 2*t.^2 + 4*t - exp(t) + 2;
% Step size
h = [0.5 0.2 0.05 0.01];
N    = length(h);
Nvec_t = 2.^(0:1:20);
hvec_t = (t(2) - t(1))./Nvec_t;
N_t = length(hvec_t);

% Solution
% Pre-allocation
[y_rk2,t_rk2,y_rk4,t_rk4,loc_err_rk2,loc_err_rk4] = deal(cell(N,1));
[glob_err_rk2,glob_err_rk4] = deal(zeros(N,1));
[comp_t_rk2,comp_t_rk4,err_rk2_t,err_rk4_t]   = deal(zeros(N_t, 1));

for i = 1:N
    % Solution
    [t_rk2{i}, y_rk2{i}, ~, ~] = RK2(f, t, h(i), x0);
    [t_rk4{i}, y_rk4{i}, ~, ~] = RK4(f, t, h(i), x0);
    % Integration errors
    glob_err_rk2(i) = max(abs(y_rk2{i}' - u_an(t_rk2{i})));
    glob_err_rk4(i) = max(abs(y_rk4{i}' - u_an(t_rk4{i})));
    loc_err_rk2{i} = abs(y_rk2{i}' - u_an(t_rk2{i}));
    loc_err_rk4{i} = abs(y_rk4{i}' - u_an(t_rk4{i}));
end
% Time analysis
for i = 1:N_t
    [~, Y, comp_t_rk2(i), ~] = RK2(f, t, hvec_t(i), x0);
    err_rk2_t(i) = abs(u_an(t(2)) - Y(end));
    [~, Y, comp_t_rk4(i), ~] = RK4(f, t, hvec_t(i), x0);
    err_rk4_t(i) = abs(u_an(t(2)) - Y(end));
end


% Plot
% RK2 solution
fig = figure();
markerObj = {'square', 'diamond', '^', 'v'};
line_opt = {'--', '-.', ':'};
legendStr = cell(N, 1);
T = linspace(t(1), t(2), 100);
plot(T, u_an(T), 'LineWidth', 2); grid on; hold on;
for i = 1:N
    if i == 1
        plot(t_rk2{i}, y_rk2{i}, '-s', 'LineWidth', 0.8);
    else
        plot(t_rk2{i}, y_rk2{i}, line_opt{i-1}, 'LineWidth', 0.8);
    end
    legendStr{i} = strcat('$h=', string(h(i)), '$');
end
xlabel('$t$ $[s]$', 'FontSize', 20); 
ylabel('$x$ $[-]$', 'FontSize', 20);
legendStr{end+1} = '$x_{ex}$';
ax = gca; ax.FontSize = 15;
legend({legendStr{end} legendStr{1:end-1}}, 'FontSize', 14, 'Location', 'best');
%save_fig(fig,'ex2_1');

% RK4 solution
fig = figure();
line_opt = {'--', '-.', ':'};
legendStr = cell(N, 1);
T = linspace(t(1), t(2), 100);
plot(T, u_an(T), 'LineWidth', 2); 
grid on; 
hold on;
for i = 1:N
    if i == 1
        plot(t_rk4{i}, y_rk4{i}, '-s', 'LineWidth', 0.8);
    else
        plot(t_rk4{i}, y_rk4{i}, line_opt{i-1}, 'LineWidth', 0.8);
    end
    legendStr{i} = strcat('$h=', string(h(i)), '$');
end
xlabel('$t$ $[s]$', 'FontSize', 20); 
ylabel('$x$ $[-]$', 'FontSize', 20);
legendStr{end+1} = '$x_{ex}$';
ax = gca; 
ax.FontSize = 15;
legend({legendStr{end} legendStr{1:end-1}}, 'FontSize', 14, 'Location', 'best');
legendStr = legendStr(1:end-1);
%save_fig(fig,'ex2_2');

% Local integration errors of RK2
fig = figure();
for i = 1:N
    if i == N
        semilogy(t_rk2{i}, loc_err_rk2{i}); hold on; grid on;
    else
        semilogy(t_rk2{i}, loc_err_rk2{i}, strcat('-', markerObj{i})); hold on; grid on;
    end
end
ax = gca; 
ax.FontSize = 13;
title('RK2: local integration error', 'FontSize', 17);
xlabel('$t [s]$', 'FontSize', 18); 
ylabel('$|x_{RK2}(t)-x(t)|$ $[-]$', 'FontSize', 18);
legend(legendStr, 'FontSize', 14, 'Location', 'best');
%save_fig(fig,'ex2_3');

% Local integration errors of RK4
fig = figure();
for i = 1:N
    if i == N
        semilogy(t_rk4{i}, loc_err_rk4{i}); hold on; grid on;
    else
        semilogy(t_rk4{i}, loc_err_rk4{i}, strcat('-', markerObj{i})); hold on; grid on;
    end
end
ax = gca; ax.FontSize = 13;
title('RK4: local integration error', 'FontSize', 17);
xlabel('$t [s]$', 'FontSize', 18); 
ylabel('$|x_{RK4}(t)-x(t)|$', 'FontSize', 18);
legend(legendStr, 'FontSize', 14, 'Location', 'southeast');
%save_fig(fig,'ex2_4');

% Global integration errors of RK2 and RK4
fig = figure();
loglog(h, glob_err_rk2, '-s'); grid on; hold on;
loglog(h, glob_err_rk4, '-^');
loglog(h, h.^2, '--');
loglog(h, 0.09*h.^4, '-.');
ax = gca; ax.FontSize = 13;
title('Global integration error', 'FontSize', 17);
xlabel('$h$ $[-]$', 'FontSize', 18); 
ylabel('$||\mbox{\boldmath $x$}_{ex}-\mbox{\boldmath $x$}||_{\infty}$ $[-]$', 'FontSize', 18);
legend({'RK2', 'RK4', '$h^2$', '$h^4$'}, 'FontSize', 14, 'Location', 'best');
%save_fig(fig,'ex2_5');

% Stochastic analysis made upon time
fig = figure();
loglog(err_rk2_t(3:end), comp_t_rk2(3:end)*1000, 'o'); 
hold on; 
grid on;
loglog(err_rk4_t(3:end), comp_t_rk4(3:end)*1000, 's');
ax = gca; 
ax.FontSize = 13;
title('Error vs Time', 'FontSize', 17);
xlabel('$||\mbox{\boldmath $x$}_{ex}(2)-\mbox{\boldmath $x$}(2)||$ $[-]$', 'FontSize', 18);
ylabel('$CPU-time$ $[ms]$', 'FontSize', 18);
legend({'RK2', 'RK4'}, 'FontSize', 14, 'Location', 'best');
%save_fig(fig,'ex2_6');
%system('cd ../Report && pdflatex main.tex > NUL');

%% EX 3
clearvars; close all; clc

% Statement of the problem
N     = 1000;
alpha = pi;
toll  = 1e-10;

% RK2 solution
A = @(alpha) [0 1; -1 2*cos(alpha)];

FRK2 = @(h,alpha) eye(2) + h*A(alpha) + 0.5*h^2*A(alpha)^2;

fun_RK2 = @(h) max(abs(eig(FRK2(h, alpha)))) - 1;
[hrk2] = secant(fun_RK2, 3, 6, toll);

% RK4 solution
FRK4 = @(h,alpha) eye(2) + h*A(alpha) + 0.5*h^2*A(alpha)^2 +...
    1/6*h^3*A(alpha)^3 + 1/24*h^4*A(alpha)^4;

fun_RK4 = @(h) max(abs(eig(FRK4(h, alpha)))) - 1;
[hrk4] = secant(fun_RK4, 3, 6, toll);

% Alpha vector [0,pi] used to create and plot the solutions.
alpha_v = [linspace(0, 80*pi/180, 1/5*N), linspace(80*pi/180, pi/2, 3/5*N),...
    linspace(pi/2, pi, 1/5*N)];

h_rk2  = zeros(N, 1);
r_rk2 = zeros(N, 1);
A = @(alpha) [0 1; -1 2*cos(alpha)];
it = 1;
for alpha = alpha_v
    fun = @(h) max(abs(eig(FRK2(h, alpha)))) - 1;
    lambda = eig(A(alpha));
    [h_rk2(it)] = secant(fun, 3, 6, toll);
    r_rk2(it) = h_rk2(it)*lambda(1);
    it = it + 1;
end


% It's not possible to find the complete solution in the rhp with the same
% method as rk2, it's necessary to search for multiple solution and order
% them to show them correcty.
h_max = 3;
corr = 2;
[r_rk4,alphaU] = solveStabilityRegion(FRK4, alpha_v, A, h_max, corr);

%%% -------------------------------- PLOTS --------------------------------

fprintf('EXERCISE 3\n\n');
fprintf('Solutions (h): \n');
fprintf('RK2:   [%2.15e]  \n', hrk2);
fprintf('RK4:   [%2.15e]  \n\n', hrk4);

% h solutions
fig = figure();
%index = find(alpha <= 0);
plot(alpha_v*180/pi, h_rk2, '-.', 'LineWidth', 1.5); grid on; hold on;
plot(alphaU*180/pi,abs(r_rk4), 'LineWidth', 1.5);
xlabel('$\alpha$ $[deg]$', 'FontSize', 18); 
ylabel('$h$ $[-]$', 'FontSize', 18);
legend({'RK2', 'RK4'}, 'FontSize', 14, 'Location', 'best');
title('Solution of the statement', 'FontSize', 17);
%save_fig(fig,'ex3_1');
clear fig

% Stable domains
figure();
re = real(r_rk2);
im = imag(r_rk2);
fp(1) = plot(re, im, '-.', 'color', [0 0.45 0.74], 'LineWidth', 1.5); hold on; grid on;
im = -im;
plot(re, im, '-.', 'color', [0 0.45 0.74], 'LineWidth', 1.5);
re = real(r_rk4);
im = imag(r_rk4);
fp(2) = plot(re, im, 'color', [0.85 0.33 0.10], 'LineWidth', 1.5); hold on; grid on;
im = -im;
plot(re, im, 'color', [0.85 0.33 0.10], 'LineWidth', 1.5);
axis equal; 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
legend(fp,{'RK2','RK4'},'FontSize', 11, 'Location', 'best');
title('Stability regions', 'FontSize', 17);
%save_fig(fig,'ex3_2');
%system('cd ../Report && pdflatex main.tex > NUL');


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
line_opt = {'-.', '--', ':'};
for i = 1:length(tolVec)
    re = real(r_rk1{i});
    im = imag(r_rk1{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    if i == 1
        fp(i) = plot(reL, imL, 'LineWidth', 1.5); hold on; grid on;
    else
        fp(i) = plot(reL, imL, line_opt{i-1}, 'LineWidth', 1.5);
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
        fp(i) = plot(reL, imL, line_opt{i-1}, 'LineWidth', 1.5);
    end
end
xlim([-2e-5 3e-5]); 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
%save_fig(fig,'ex4_1');


% RK2 solutions
fig = figure();
line_opt = {'-.', '--', ':'};
for i = 1:length(tolVec)
    re = real(r_rk2{i});
    im = imag(r_rk2{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    if i == 1
        fp(i) = plot(reL, imL, 'LineWidth', 1.5); hold on; grid on;
    else
        fp(i) = plot(reL, imL, line_opt{i-1}, 'LineWidth', 1.5);
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
line_opt = {'-.', '--', ':'};
for i = 1:length(tolVec)
    re = real(r_rk4{i});
    im = imag(r_rk4{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    if i == 1
        fp(i) = plot(reL, imL, 'LineWidth', 1.5); hold on; grid on;
    else
        fp(i) = plot(reL, imL, line_opt{i-1}, 'LineWidth', 1.5);
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
loglog(tolVec, feval1, '-s', 'LineWidth', 1.5); 
hold on;
loglog(tolVec, feval2, '-^', 'LineWidth', 1.5)
loglog(tolVec, feval4, '-o', 'LineWidth', 1.5); 
grid on;
title('f evaluations vs Tolerance', 'FontSize', 17);
xlabel('tol [-]', 'FontSize', 18); 
ylabel('f evaluations [-]', 'FontSize', 18);
legend({'RK1', 'RK2', 'RK4'}, 'FontSize', 14, 'Location', 'best');
%save_fig(fig,'ex4_4');
%system('cd ../Report && pdflatex main.tex > NUL');

%% EX 5
clearvars; close all; clc; warning off;

% Statement of the problem
A = @(alpha) [0 1; -1 2*cos(alpha)];
alpha_v = linspace(0, pi, 1000);
theta_v = [0.4 0.1 0.3 0.7 0.9];

% Linear operator derived function of h alpha and theta
F_BI = @(h, a, th) (eye(2) - (1 - th)*h*A(a) + ...
    ((((1 - th)*h)^2)/2)*A(a)^2)\(eye(2) + th*h*A(a) + (((th*h)^2)/2)*A(a)^2);

[h_v,R_v,re,im] = deal(zeros(length(alpha_v),length(theta_v)));
[reL, imL] = deal(zeros(length(alpha_v)*2,length(theta_v)));
[alphaL, RL] = deal(cell(length(theta_v), 1));


% Calculations for every theta saving the result in a cell for each theta
% in order to be plotted
for i = 1:length(theta_v)
    for j = 1:length(alpha_v)
        fun = @(h) max(abs(eig(F_BI(h, alpha_v(j), theta_v(i)))))-1;  
        h_v(j,i) = fzero(fun, 9, optimset('display', 'none'));
        lambda = eig(A(alpha_v(j)));   
        R_v(j,i) = h_v(j,i)*lambda(1);
    end
    re = real(R_v(:,i));
    im = imag(R_v(:,i));
    reL(:,i) = [re; re(end:-1:1)];
    imL(:,i) = [im; -im(end:-1:1)];
    [alpha, indA, ~] = unique(atan2d(im, re));
    ind = find(alpha <= 90, 1, 'last');
    R = R_v(indA,i);
    R = [R(1:ind); 0; R(ind+1:end)];
    RL{i} = R;
    alphaL{i} = [alpha(1:ind); 90; alpha(ind+1:end)];
end

%%% -------------------------------- PLOTS --------------------------------
% BI2_theta unstable domains
fig = figure();
hold on; 
grid on;
plot(reL, imL, 'LineWidth', 1.5);
x_lim = xlim; 
xlim([x_lim(1)-0.5 x_lim(2)+0.5]);
axis equal; 
ax = gca; 
set(ax,'box','on')
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend({'$\theta=0.4$','$\theta=0.1$','$\theta=0.3$','$\theta=0.7$','$\theta=0.9$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
title('Unstable regions of $BI2_{\theta}$', 'FontSize', 17);
save_fig(fig,'ex5_1');

% h solutions
fig = figure();
hold on; 
grid on;
for i = 1:length(theta_v)
   plot(alphaL{i}, abs(RL{i}),'LineWidth', 1.5);
end
ax = gca;
set(ax,'box','on')
xlim([0 180]);
legend({'$\theta=0.4$','$\theta=0.1$','$\theta=0.3$','$\theta=0.7$','$\theta=0.9$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$\alpha$ [deg]', 'FontSize', 18); 
ylabel('$h$ [-]', 'FontSize', 18);
title('Minimum $h$ for stability', 'FontSize', 17);
save_fig(fig,'ex5_2');
%system('cd ../Report && pdflatex main.tex > NUL');

%% EX 6
clearvars; close all; clc

% Statement of the problem
x0 = [1 1]';
t_int = [0 5];
% Analytical solution
B = [-180.5 219.5; 179.5 -220.5];
eig_B = eig(B)*0.1;
x_an = @(t) expm(t*B)*x0;
f = @(t, x) B*x;
% RK4 solution
[t_rk4, y_rk4, ~, ~] = RK4(f, t_int, 0.1, x0);
N = 1000;
A = @(alpha) [0 1; -1 2*cos(alpha)];
% ------------------------- Solution with IEX4 --------------------------
% step
h = 0.1;
t_v = t_int(1):h:t_int(2);
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
plot(eig_B(2), 0, 'ko', 'markerSize', 10);
x_lim = xlim; 
xlim([x_lim(1)-0.5 x_lim(2)+0.5]);
y_lim = ylim; 
ylim([y_lim(1)-0.5 y_lim(2)+0.5]);
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
save_fig(fig,'ex6_1');

% RK4 solution
fig =  figure();
plot(t_rk4, y_rk4(:, 1), 'LineWidth', 1.5); 
grid on; 
hold on;
plot(t_rk4, y_rk4(:, 2), '--', 'LineWidth', 1.5);
legend({'$x_1$', '$x_2$'}, 'FontSize', 16, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$x$ [-]', 'FontSize', 18);
title('RK4 - solution', 'FontSize', 17);
save_fig(fig,'ex6_2');

% IEX4 solution
fig = figure();
plot(t_rk4, y_IEX4(1, :), 'LineWidth', 1.5); 
grid on; 
hold on;
plot(t_rk4, y_IEX4(2, :), '--', 'LineWidth', 1.5);
legend({'$x_1$', '$x_2$'}, 'FontSize', 16, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$x$ [-]', 'FontSize', 18);
title('IEX4 - solution', 'FontSize', 17);
save_fig(fig,'ex6_4');

% RK4 local integration error
fig = figure();
semilogy(t_v, abs(y_an(1, :) - y_rk4(:, 1)'), 'LineWidth', 1.5);
grid on; 
hold on;
semilogy(t_v, abs(y_an(2, :) - y_rk4(:, 2)'), '--', 'LineWidth', 1.5);
legend({'Error on first component', 'Error on second component'},...
    'FontSize', 16, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('Error [-]', 'FontSize', 18);
title('Error between analytical and RK4 method', 'FontSize', 17);
save_fig(fig,'ex6_3');

% IEX4 local integration error
fig = figure();
semilogy(t_v, abs(y_an(1, :) - y_IEX4(1, :)), 'LineWidth', 1.5); 
grid on; 
hold on;
semilogy(t_v, abs(y_an(2, :) - y_IEX4(2, :)), '--', 'LineWidth', 1.5);
legend({'Error on first component', 'Error on second component'},...
    'FontSize', 16, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('Error [-]', 'FontSize', 18);
title('Error between analytical and IEX4 method', 'FontSize', 17);
save_fig(fig,'ex6_5');
%system('cd ../Report && pdflatex main.tex > NUL');

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
alpha_v = [linspace(0, 80*pi/180, 1/5*N),...
             linspace(80*pi/180, pi/2, 3/5*N),...
                linspace(pi/2, pi, 1/5*N)];

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
[R_BDF3,~] = solveStabilityRegion(F_BDF3, alpha_v, A, 6, -6);
R_BDF3 = R_BDF3([1,3:end]); %remove a zero that creates problem to the plots
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
legend('$x_1$', '$x_2$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
% save_fig(fig,'ex7_1');

% AM3 solution
fig = figure('Name', 'EX 7 - AM3', 'NumberTitle', 'off');
plot(t_AM3, Y_AM3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AM3, Y_AM3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('AM3', 'FontSize', 17);
legend('$x_1$', '$x_2$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
% save_fig(fig,'ex7_2');

% ABM3 solution
fig = figure('Name', 'EX 7 - ABM3', 'NumberTitle', 'off');
plot(t_ABM3, Y_ABM3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_ABM3, Y_ABM3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('ABM3', 'FontSize', 17);
legend('$x_1$', '$x_2$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
a2 = axes();
a2.Position = [0.19 0.19 0.4000 0.2500];
box on
hold on; 
grid on;
plot(t_ABM3, Y_ABM3(:, 1),'-.');
ax = gca;
ax.XLim = [1  2.5];
ax.YLim = [-0.2 0.2];
save_fig(fig,'ex7_3');

% BDF3 solution
fig = figure('Name', 'EX 7 - BDF3', 'NumberTitle', 'off');
plot(t_BDF3, Y_BDF3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_BDF3, Y_BDF3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('BDF3', 'FontSize', 17);
legend('$x_1$', '$x_2$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
% save_fig(fig,'ex7_4');

% Real system eigenvalues
M = @(t) [-5/2*(1 + 8*sin(t)) 0; 1 1];
lambda = zeros(2, length(t_AB3));
for i = 1:length(t_AB3)
    lambda(:, i) = eig(M(t_AB3(i)))*0.1;
end
fig = figure('Name', 'EX 7 - Real system eigenvalues', 'NumberTitle', 'off');
plot(t_AB3, lambda(2, :), 'LineWidth', 1.5); 
grid on; 
hold on;
plot(t_AB3, lambda(1, :), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$h\lambda_i$ [-]', 'FontSize', 18);
title('$h\lambda_i$ of the linearized system', 'FontSize', 17);
legend('$\lambda_{x_1}$', '$\lambda_{x_2}$', 'FontSize', 14, 'Location','best');
% save_fig(fig,'ex7_5');

% Stability/Instability domains
fig = figure('Name', 'EX 7 - Stability region', 'NumberTitle', 'off');
plot(reTot_AB3, imTot_AB3, 'LineWIdth', 1.5); 
hold on; 
grid on;
plot(reTot_AM3, imTot_AM3, '-.', 'LineWidth', 1.5);
plot(reTot_ABM3, imTot_ABM3, ':', 'LineWidth', 1.5);
plot(reTot_BDF3, imTot_BDF3);
line([real(min(lambda(2,:))) real(max(lambda(2,:)))], [0 0], 'color', 'red', 'LineWidth', 2);
plot(mean(lambda(1,:)), 0, 'ko', 'markerSize', 5);
x_lim = xlim; 
xlim([x_lim(1)-0.5 x_lim(2)+0.1]);
y_lim = ylim; 
ylim([y_lim(1)-0.5 y_lim(2)+0.5]);
axis equal; 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend({'AB3 Stability', 'AM3 Stability',...
    'ABM3 Stability', 'BDF3 Instability','$\lambda_{x1}$','$\lambda_{x2}$'},...
    'FontSize', 10, 'Location', 'southeast');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
a2 = axes();
a2.Position = [0.56 0.7 0.31 0.21];
box on
hold on; 
grid on;
plot(reTot_AB3, imTot_AB3, 'LineWIdth', 1.5); 
plot(reTot_AM3, imTot_AM3, '-.', 'LineWidth', 1.5);
plot(reTot_ABM3, imTot_ABM3, ':', 'LineWidth', 1.5);
plot(reTot_BDF3, imTot_BDF3);
plot(mean(lambda(1,:)), 0, 'ko', 'markerSize', 5);
line([real(min(lambda(2,:))) real(max(lambda(2,:)))], [0 0], 'color', 'red', 'LineWidth', 2);
ax = gca;
ax.XLim = [-2.5  0.5];
ax.YLim = [-0.4 0.4];
save_fig(fig,'ex7_6');
%system('cd ../Report && pdflatex main.tex');

%% Functions

% All the functions have an intestation with the description of the
% function and the definition of inputs and outputs. It's possible to use
% the "help" command to retrieve informations about each function
%
% Most of the functions return the time of execution, this is done inside 
% of the function in order to evaluate only the algorithm and execlude time
% used for the function call. This was used during the design phase
% together with MATLAB Profiler app.

%%% ------------------------------ EX-1 ---------------------------------

function [sol, it] = newtAn(f, x, invJ, toll)
% newtAn - Computes the zero of a multidimensional function using Newton's method.
%
%   [t, sol, it, feval] = newtAn(f, x, invJ, toll)
%
% INPUT:
%   f        - Function handle [1x1] for evaluating the objective function.
%   x        - Double [nx1] representing the starting guess.
%   invJ     - Function handle [1x1] for the analytic inverse Jacobian matrix.
%   toll     - Double [1x1] specifying the tolerance for convergence.
%
% OUTPUT:
%   t        - Double [1x1] representing the computational time needed in seconds.
%   sol      - Double [nx1] representing the solution.
%   it       - Double [1x1] indicating the number of iterations.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   newtAn computes the zero of a multidimensional function using Newton's
%   method. The function iteratively updates the solution based on the
%   analytic inverse Jacobian matrix until the convergence tolerance is met.
%         
% -------------------------------------------------------------------------

% The initial error needs to be greater then the toll
err = toll + 1; 
it = 0;
f_x = f(x);
% any() is necessary since f_x is a 2x1 element
while any(err > toll)
    x = x - invJ(x) * f_x;
    f_x = f(x);
    err = abs(f_x);
    it = it + 1;
end
sol = x;

end


function [sol, it] = newtFF(f, x, toll)
% newtFF - Computes the zero of a multidimensional function using Newton's method
%                with finite forward differences for the Jacobian matrix.
%
%   [t, sol, it, feval] = newtFF(f, x, toll)
%
% INPUT:
%   f        - Function handle [1x1] for evaluating the objective function.
%   x        - Double [nx1] representing the starting guess.
%   toll     - Double [1x1] specifying the tolerance for convergence.
%
% OUTPUT:
%   sol      - Double [nx1] representing the solution.
%   it       - Double [1x1] indicating the number of iterations.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   newtFF computes the zero of a multidimensional function using
%   Newton's method. The Jacobian matrix is approximated using finite forward
%   differences. The function iteratively updates the solution until the
%   convergence tolerance is met.
%
% -------------------------------------------------------------------------
% The initial error needs to be greater then the toll
err = toll + 1;
it = 0;
f_x = f(x);

while any(err > 1e-8)
    J = zeros(2);
    delta = sqrt(eps).*min(abs(x), 1);

    J(:, 1) = ((f([x(1)+delta(1), x(2)]) - f_x)/delta(1));
    J(:, 2) = ((f([x(1), x(2)+delta(2)]) - f_x)/delta(2));
    
    x = x - J\f_x;
    f_x = f(x);
    err = abs(f_x);
    it = it + 1;
    
end

sol = x;

end


function [sol, it] = newtFC(f, x, toll)
% newtFC - Computes the zero of a multidimensional function using 
%                 Newton's method with centered finite differences for the 
%                 Jacobian matrix.
%
%   [t, sol, it, feval] = newtFC(f, x, toll)
%
% INPUT:
%   f        - Function handle [1x1] for evaluating the objective function.
%   x        - Double [nx1] representing the starting guess.
%   toll     - Double [1x1] specifying the tolerance for convergence.
%
% OUTPUT:
%   t        - Double [1x1] computational time needed in seconds.
%   sol      - Double [nx1] representing the solution.
%   it       - Double [1x1] indicating the number of iterations.
%   feval     - Double [1x1] representing the number of function evaluations.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   newtFC computes the zero of a multidimensional function using
%   Newton's method. The Jacobian matrix is approximated using centered
%   finite differences. The function iteratively updates the solution until
%   the convergence tolerance is met.
%
% -------------------------------------------------------------------------

% The initial error needs to be greater then the toll
err = toll + 1;
it = 0;
f_x = f(x);

while any(err > 1e-8)
    J = zeros(2);
    delta = sqrt(eps).* min(abs(x), 1);

    J(:, 1) = ((f([x(1)+delta(1), x(2)]) - f([x(1)-delta(1), x(2)]))/(2*delta(1)));
    J(:, 2) = ((f([x(1), x(2)+delta(2)]) - f([x(1), x(2)-delta(2)]))/(2*delta(2)));
    
    x = x - J\f_x;
    f_x = f(x);
    err = abs(f_x);
    it = it + 1;
    
end
sol = x;

end

function [sol] = secant(f, a, b, toll)
% secant - Retrieve the solution of finding zero using the secant method.
%
%   [t, sol, it, feval, cV] = secant(f, a, b, toll)
%
% INPUT:
%   f        - Function handle [1x1] for evaluating the objective function.
%   a        - Double [1x1] representing the left end of the zero searching range.
%   b        - Double [1x1] representing the right end of the zero searching range.
%   toll     - Double [1x1] specifying the tolerance for convergence.
%
% OUTPUT:
%   sol      - Double [1x1] representing the solution.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   secant retrieves the solution of finding zero using the secant
%   method. The function iteratively updates the solution based on the secant
%   approximation until the convergence tolerance is met.
%
% -------------------------------------------------------------------------


err = toll + 1;
f1 = f(a);
f2 = f(b);

while err > toll
    c = b - (f2 * (b - a))/(f2 - f1);
    fc = f(c);
    err = abs(fc); 
    a = b;
    b = c;
    f1 = f2;
    f2 = fc;
    sol = c;
end

end


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

N = round(diff(t_int)/h);
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

N = round(diff(t_int)/h);
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
k2 = h / 12 * [5; 8; -1];

% First 2 steps are obtained with RK4 method
[~, y, ~, ~] = RK4(f, [0 3*h], h, x0);

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

function save_fig(fig,name)
% save_fig - Function to save a MATLAB figure to a PDF file with correct sizing.
%
% PROTOTYPE
%   save_fig(fig, name)
%
% INPUT:
%   fig      - Figure [1x1] to be saved.
%   name     - String [1xN] specifying the desired name of the saved file.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   save_fig saves the given figure to a PDF file with correct sizing. It
%   retrieves the figure's aspect ratio, sets appropriate paper size, and
%   automatically adjusts the paper position. The saved file is stored in
%   the 'Report/gfx/' directory with the specified name.
%
% -------------------------------------------------------------------------

WHratio = fig.Position(3)/fig.Position(4); % retrieve current WHratio
widthPos = 15;
heightPos = widthPos/WHratio;

set(fig,'Units','centimeters',...
       'PaperUnits','centimeters',...
       'PaperSize',[widthPos heightPos],...
       'PaperPositionMode','auto',...
       'InvertHardcopy', 'on');
name = strcat('..\Report\gfx\',name);
saveas(fig,name,'pdf')
close(fig)
end