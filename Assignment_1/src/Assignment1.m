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

%% Ex 1
clearvars; close all; clc

% Function to evaluate
func = @(x) [x(2)^2 - x(1) - 2; -x(1)^2 + x(2) + 10];
% Inverse Jacobian of the function
invJ = @(x) (1/(4*x(1)*x(2)-1))*[1, -2*x(2); 2*x(1), -1];
% Tolerance
toll = 1e-8;  

% Finding the first zero z1        
x0 = [1 -4]';
% Analytical method, netwon finite forward, newton finite centered
[sol1a, n1a] = newtAn(func, x0, invJ, toll);
[sol1f, n1f] = newtFF(func, x0, toll);
[sol1c, n1c] = newtFC(func, x0, toll);

% Finding the second zero z2
x0 = [6 5]';
% Analytical method, netwon finite forward, newton finite centered
[sol2a, n2a] = newtAn(func, x0, invJ, toll);
[sol2f, n2f] = newtFF(func, x0, toll);
[sol2c, n2c] = newtFC(func, x0, toll);

%%% ----------------------------- PRINT DATA ------------------------------
err_1f = norm(sol1a-sol1f);
err_2f = norm(sol2a-sol2f);
err_1c = norm(sol1a-sol1c);
err_2c = norm(sol2a-sol2c);

fprintf('--------------- EXERCISE 1 ---------------\n\n');
fprintf('## Solutions of first zero: \n');
fprintf('newtAn:   [%2.15f, %2.15f]  \n', sol1a);
fprintf('newtFF:   [%2.15f, %2.15f]  \n', sol1f);
fprintf('newtFC:   [%2.15f, %2.15f]  \n\n', sol1c);
fprintf('## Solutions of second zero: \n');
fprintf('newtAn:   [%2.15f, %2.15f]  \n', sol2a);
fprintf('newtFC:   [%2.15f, %2.15f]  \n', sol2f);
fprintf('newtFF:   [%2.15f, %2.15f]  \n\n', sol2c);
fprintf('## Number of iterations: \n');
fprintf('newtAn:   [%d, %d]  \n', n1a, n2a);
fprintf('newtFC:   [%d, %d]  \n', n1f, n2f);
fprintf('newtFF:   [%d, %d]  \n\n', n1c, n2c);
fprintf('## Error respect analytical zero: \n');
fprintf('newtAn:   [-, -]  \n');
fprintf('newtFC:   [%2.15e, %2.15e]  \n', err_1f, err_2f);
fprintf('newtFF:   [%2.15e, %2.15e]  \n', err_1c, err_2c);


%%% -------------------------------- PLOTS --------------------------------

f1 = @(x1, x2) x2.^2 - x1 - 2;
f2 = @(x1, x2) -x1.^2 + x2 + 10;

% Surface plot of the function f1
fig = figure('Name', 'EX 1 - F1 function', 'NumberTitle', 'off');
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
fig = figure('Name', 'EX 1 - F2 function', 'NumberTitle', 'off');
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
fig = figure('Name', 'EX 1 - Solution', 'NumberTitle', 'off');
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

%%% ------------------------------- Solution -------------------------------
% Pre-allocation
y_rk2 = cell(N, 1);
t_rk2 = cell(N, 1);
y_rk4 = cell(N, 1);
t_rk4 = cell(N, 1);
loc_err_rk2  = cell(N, 1);
glob_err_rk2 = zeros(N, 1);
loc_err_rk4  = cell(N, 1);
glob_err_rk4 = zeros(N, 1);

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


%%% ---------------------------- TIME ANALYSIS ----------------------------
Nvec_t = 2.^(0:1:20);
hvec_t = (t(2) - t(1))./Nvec_t;
N_t = length(hvec_t);
% Preallocation
comp_t_rk2   = zeros(N_t, 1);
err_rk2_t = zeros(N_t, 1);
comp_t_rk4   = zeros(N_t, 1);
err_rk4_t = zeros(N_t, 1);

% Computations
for i = 1:N_t
    [~, Y, comp_t_rk2(i), ~] = RK2(f, t, hvec_t(i), x0);
    err_rk2_t(i) = abs(u_an(t(2)) - Y(end));
    [~, Y, comp_t_rk4(i), ~] = RK4(f, t, hvec_t(i), x0);
    err_rk4_t(i) = abs(u_an(t(2)) - Y(end));
end


%%% -------------------------------- PLOTS --------------------------------
% RK2 solution
fig = figure('Name', 'EX 2 - solution RK2', 'NumberTitle', 'off');
markerObj = {'square', 'diamond', '^', 'v'};
lineSpecObj = {'--', '-.', ':'};
legendStr = cell(N, 1);
T = linspace(t(1), t(2), 100);
plot(T, u_an(T), 'LineWidth', 2); grid on; hold on;
for i = 1:N
    if i == 1
        plot(t_rk2{i}, y_rk2{i}, '-s', 'LineWidth', 0.8);
    else
        plot(t_rk2{i}, y_rk2{i}, lineSpecObj{i-1}, 'LineWidth', 0.8);
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
fig = figure('Name', 'EX 2 - solution RK4', 'NumberTitle', 'off');
lineSpecObj = {'--', '-.', ':'};
legendStr = cell(N, 1);
T = linspace(t(1), t(2), 100);
plot(T, u_an(T), 'LineWidth', 2); 
grid on; 
hold on;
for i = 1:N
    if i == 1
        plot(t_rk4{i}, y_rk4{i}, '-s', 'LineWidth', 0.8);
    else
        plot(t_rk4{i}, y_rk4{i}, lineSpecObj{i-1}, 'LineWidth', 0.8);
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
fig = figure('Name', 'EX 2 - RK2 local integration error', 'NumberTitle', 'off');
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
fig = figure('Name', 'EX 2 - RK4 local integration error', 'NumberTitle', 'off');
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
fig = figure('Name', 'EX 2 - global integration error', 'NumberTitle', 'off');
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
fig = figure('Name', 'EX 2 - ERR vs TIME', 'NumberTitle', 'off');
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

funRK2 = @(h) max(abs(eig(FRK2(h, alpha)))) - 1;
[hrk2] = secantMethod(funRK2, 3, 6, toll);

% RK4 solution
FRK4 = @(h,alpha) eye(2) + h*A(alpha) + 0.5*h^2*A(alpha)^2 +...
    1/6*h^3*A(alpha)^3 + 1/24*h^4*A(alpha)^4;

funRK4 = @(h) max(abs(eig(FRK4(h, alpha)))) - 1;
[hrk4] = secantMethod(funRK4, 3, 6, toll);

fprintf('*************** EXERCISE 3 ***************\n');
fprintf('## Solutions (h): \n');
fprintf('RK2 method:   %2.15e  \n', hrk2);
fprintf('RK4 method:   %2.15e  \n\n', hrk4);

% Alpha vector [0,pi] used to create and plot the solutions.
alphaVec = [linspace(0, 75*pi/180, 1/10*N), linspace(75*pi/180, pi/2, 8/10*N), linspace(pi/2, pi, 1/10*N)];

h_rk2  = zeros(N, 1);
r_rk2 = zeros(N, 1);
A = @(alpha) [0 1; -1 2*cos(alpha)];
it = 1;
for alpha = alphaVec
    fun = @(h) max(abs(eig(FRK2(h, alpha)))) - 1;
    lambda = eig(A(alpha));
    [h_rk2(it)] = secantMethod(fun, 3, 6, toll);
    r_rk2(it) = h_rk2(it)*lambda(1);
    it = it + 1;
end

h_max = 3;
corr = 2;
[r_rk4,alphaU] = solveStabilityRegion(FRK4, alphaVec, A, h_max, corr);

%%% -------------------------------- PLOTS --------------------------------
% h solutions
fig = figure('Name', 'EX 3 - h', 'NumberTitle', 'off');
%index = find(alpha <= 0);
plot(alphaVec*180/pi, h_rk2, '-.', 'LineWidth', 1.5); grid on; hold on;
plot(alphaU*180/pi,abs(r_rk4), 'LineWidth', 1.5);
xlabel('$\alpha$ $[deg]$', 'FontSize', 18); 
ylabel('$h$ $[-]$', 'FontSize', 18);
legend({'RK2', 'RK4'}, 'FontSize', 14, 'Location', 'best');
title('Solution of the statement', 'FontSize', 17);
%save_fig(fig,'ex3_1');
clear fig

% Stable domains
figure('Name', 'EX 3 - Regions', 'NumberTitle', 'off');
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
legend(fp,{'RK2 Stability margin','RK4 Stability margin'},...
                        'FontSize', 11, 'Location', 'best');
title('Stability regions', 'FontSize', 17);
%save_fig(fig,'ex3_2');
%system('cd ../Report && pdflatex main.tex > NUL');


%% EX 4
clearvars; close all; clc

% Statement of the problem
x0 = [1 1]';
A = @(alpha) [0 1; -1 2*cos(alpha)];
tlim = [0 1];

% Analytical solution
x_an = @(t, alpha) expm(t*A(alpha))*x0;

%%% NEEDED FUNCTIONS ------------------------------------------------------
% function to find h and N values for the ode functions
getN = @(h) nth_output(1, @getNH, h, tlim);
getHbar = @(h) nth_output(2, @getNH, h, tlim);

% Linear operator of the methods
F_RK1 = @(h, alpha) eye(length(x0)) + h*A(alpha);
F_RK2 = @(h, alpha) eye(length(x0)) + h*A(alpha) + (h^2)/2*A(alpha)^2;
F_RK4 = @(h, alpha) eye(length(x0)) + h*A(alpha) + (h^2)/2*A(alpha)^2 ...
                            + ((h^3)/6)*A(alpha)^3 + ((h^4)/24)*A(alpha)^4;

% Solution functions with the three methods
x_RK1 = @(h, alpha) F_RK1(getHbar(h), alpha)*(F_RK1(h, alpha)^getN(h))*x0;
x_RK2 = @(h, alpha) F_RK2(getHbar(h), alpha)*(F_RK2(h, alpha)^getN(h))*x0;
x_RK4 = @(h, alpha) F_RK4(getHbar(h), alpha)*(F_RK4(h, alpha)^getN(h))*x0;

% Functions to be solved
fun1 = @(h, alpha, toll) norm(x_an(tlim(2), alpha) - x_RK1(h, alpha), 'inf') - toll;
fun2 = @(h, alpha, toll) norm(x_an(tlim(2), alpha) - x_RK2(h, alpha), 'inf') - toll;
fun4 = @(h, alpha, toll) norm(x_an(tlim(2), alpha) - x_RK4(h, alpha), 'inf') - toll;

%%% CALCULATIONS ----------------------------------------------------------
% Pre-allocation
alphaVec = linspace(0, pi, 100);
tolVec = [1e-3, 1e-4, 1e-5, 1e-6];

h_rk1 = cell(length(tolVec), 1);
h_rk2 = cell(length(tolVec), 1);
h_rk4 = cell(length(tolVec), 1);

R_RK1 = cell(length(tolVec), 1);
r_rk2 = cell(length(tolVec), 1);
R_RK4 = cell(length(tolVec), 1);

fcnE1 = zeros(length(tolVec), 1);
fcnE2 = zeros(length(tolVec), 1);
fcnE4 = zeros(length(tolVec), 1);

opt = optimoptions('fsolve', 'Display', 'none');

% Loop to find all the h values that zeroed the functions fun(i)
for i = 1:length(tolVec)
    toll = tolVec(i);
    h_rk1{i} = zeros(length(alphaVec), 1);
    h_rk2{i} = zeros(length(alphaVec), 1);
    h_rk4{i} = zeros(length(alphaVec), 1);

    R_RK1{i} = zeros(length(alphaVec), 1);
    r_rk2{i} = zeros(length(alphaVec), 1);
    R_RK4{i} = zeros(length(alphaVec), 1);

    F11 = @(h, alpha) fun1(h, alpha, toll);
    F12 = @(h, alpha) fun2(h, alpha, toll);
    F14 = @(h, alpha) fun4(h, RK2(alpha, toll));
    
    for j = 1:length(alphaVec)
        F21 = @(h) F11(h, alphaVec(j));
        F22 = @(h) F12(h, alphaVec(j));
        F24 = @(h) F14(h, alphaVec(j));
        
        h_rk1{i}(j) = fsolve(F21, 5e-5, opt);
        h_rk2{i}(j) = fzero(F22, [5e-6, 1]);
        h_rk4{i}(j) = fzero(F24, [5e-6, 1]);
        
        lambda = eig(A(alphaVec(j))); 
        lambda = lambda(1);
        R_RK1{i}(j) = h_rk1{i}(j)*lambda;
        r_rk2{i}(j) = h_rk2{i}(j)*lambda;
        R_RK4{i}(j) = h_rk4{i}(j)*lambda;
    end
    
end

% Obtain function evaluations number
for i = 1:length(tolVec)
    toll = tolVec(i);
    [~, ~, ~, fcnE1(i)] = RK1(@(t, x) A(pi)*x, tlim, h_rk1{i}(end), x0);
    [~, ~, ~, fcnE2(i)] = RK2(@(t, x) A(pi)*x, tlim, h_rk2{i}(end), x0);
    [~, ~, ~, fcnE4(i)] = RK4(@(t, x) A(pi)*x, tlim, h_rk4{i}(end), x0);
end

%%% -------------------------------- PLOTS --------------------------------
% RK1 solutions
fig = figure('Name', 'EX 4 - RK1', 'NumberTitle', 'off');
lineSpecObj = {'-.', '--', ':'};
for i = 1:length(tolVec)
    re = real(R_RK1{i});
    im = imag(R_RK1{i});
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
    re = real(R_RK1{i});
    im = imag(R_RK1{i});
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
fig = figure('Name', 'EX 4 - RK2', 'NumberTitle', 'off');
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
fig = figure('Name', 'EX 4 - RK4', 'NumberTitle', 'off');
lineSpecObj = {'-.', '--', ':'};
for i = 1:length(tolVec)
    re = real(R_RK4{i});
    im = imag(R_RK4{i});
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
fig = figure('Name', 'EX 4 - f evaluations', 'NumberTitle', 'off');
loglog(tolVec, fcnE1, '-s', 'LineWidth', 1.5); hold on;
loglog(tolVec, fcnE2, '-^', 'LineWidth', 1.5)
loglog(tolVec, fcnE4, '-o', 'LineWidth', 1.5); grid on;
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
alphaVec = linspace(0, pi, 1000);
thetaVec = [0.4 0.1 0.3 0.7 0.9];

% Linear operator derived function of h alpha and theta
F_BI = @(h, a, th) (eye(2) - (1 - th)*h*A(a) + ...
    ((((1 - th)*h)^2)/2)*A(a)^2)\(eye(2) + th*h*A(a) + (((th*h)^2)/2)*A(a)^2);

% Pre-allocation
h = cell(length(thetaVec), 1);
R = cell(length(thetaVec), 1);

% fzero options, remove the warnings and info to clean the output
opts = optimset('display', 'none');
% Calculations for every theta saving the result in a cell for each theta
for i = 1:length(thetaVec)
    h{i} = zeros(length(alphaVec), 1);
    R{i} = zeros(length(alphaVec), 1);
    for j = 1:length(alphaVec)
        fun = @(h) max(abs(eig(F_BI(h, alphaVec(j), thetaVec(i)))))-1;  
        h{i}(j) = fzero(fun, 9, opts);
        lambda = eig(A(alphaVec(j))); 
        R{i}(j) = h{i}(j)*lambda(1);     
        
    end
end



%%% -------------------------------- PLOTS --------------------------------
% BI2_theta unstable domains
fig = figure('Name', 'EX 5 - BI2', 'NumberTitle', 'off');
str = {''};
hold on; 
grid on;
for i = 1:length(thetaVec)
    re = real(R{i});
    im = imag(R{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    fp(i) = plot(reL, imL, 'LineWidth', 1.5);
    str{i} = strcat('$\theta$', " = ", '$', string(thetaVec(i)),'$');
end
xLIM = xlim; 
xlim([xLIM(1)-0.5 xLIM(2)+0.5]);
axis equal; 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend(fp, str, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
title('Unstable regions of $BI2_{\theta}$', 'FontSize', 17);
% save_fig(fig,'ex5_1');

% h solutions
fig = figure('Name', 'EX 5 - BI2 h', 'NumberTitle', 'off');
hold on; grid on;
for i = 1:length(thetaVec)
    re = real(R{i});
    im = imag(R{i});
    alpha = atan2d(im, re);
    [alpha, IA, IC] = unique(alpha);
    R{i} = R{i}(IA);
    ind = find(alpha <= 90, 1, 'last');
    R{i} = [R{i}(1:ind); 0; R{i}(ind+1:end)];
    alpha = [alpha(1:ind); 90; alpha(ind+1:end)];
    fp(i) = plot(alpha, abs(R{i}),'LineWidth', 1.5);

end
xlim([0 180]);
legend(fp, str, 'FontSize', 14, 'Location', 'best');
xlabel('$\alpha$ [deg]', 'FontSize', 18); 
ylabel('$h$ [-]', 'FontSize', 18);
title('Minimum $h$ for stability', 'FontSize', 17);
% save_fig(fig,'ex5_2');
system('cd ../Report && pdflatex main.tex > NUL');

%% EX 6
clearvars; close all; clc

% Statement of the problem
B = [-180.5 219.5; 179.5 -220.5];
x0 = [1 1]';
tlim = [0 5];
% Analytical solution
x_an = @(t) expm(t*B)*x0;

f = @(t, x) B*x;
% RK4 solution
[t_rk4, y_rk4, ~, ~] = RK4(f, tlim, 0.1, x0);

%%%  ----------------------- RK4 stability domain -----------------------
% The same code to generate the stability domain is taken from Exercise 3 
N = 1000;
A = @(alpha) [0 1; -1 2*cos(alpha)];
alphaVec = linspace(0,pi,N);
h_max = 3;
corr = 2;
FRK4 = @(h,alpha) eye(2) + h*A(alpha) + 0.5*h^2*A(alpha)^2 +...
    1/6*h^3*A(alpha)^3 + 1/24*h^4*A(alpha)^4;
[R_RK4,~] = solveStabilityRegion(FRK4, alphaVec, A, h_max, corr);

%  ----------------------- IEX4 instability domain ----------------------
alphaVec = linspace(0, pi, 500);
hIEX4  = zeros(length(alphaVec), 1);
R_IEX4 = zeros(length(alphaVec), 1);
A = @(alpha) [0 1; -1 2*cos(alpha)];

FIEX4 = @(h, alpha) (-1/6)*((eye(2)-h*A(alpha)))^(-1)+4*(((eye(2)-(h/2)*A(alpha))^2))^(-1)-...
    (27/2)*(((eye(2)-(h/3)*A(alpha))^3))^(-1)+(32/3)*(((eye(2)-(h/4)*A(alpha))^4))^(-1); 

% Definition of a guess 
hGuess = 10;
it = 1;

for alpha = alphaVec
    fun = @(h) max(abs(eig(FIEX4(h, alpha)))) - 1;
    hIEX4(it) = fzero(fun, hGuess);
    hGuess = hIEX4(it);
    lambda = eig(A(alpha)); lambda = lambda(1);
    R_IEX4(it) = hIEX4(it)*lambda;
    it = it + 1;
end

% ------------------------- Solution with IEX4 --------------------------
B = [-180.5 219.5; 179.5 -220.5];
h = 0.1;
tVec = tlim(1):h:tlim(2);
Y_IEX4 = [x0, zeros(2, length(tVec)-1)];
Y_an = [x0, zeros(2, length(tVec)-1)];

F_IEX4 = @(h) (-1/6)*((eye(2)-h*B))^(-1)+4*(((eye(2)-(h/2)*B)^2))^(-1)-...
    (27/2)*(((eye(2)-(h/3)*B)^3))^(-1)+(32/3)*(((eye(2)-(h/4)*B)^4))^(-1); 

for i = 2:length(tVec)
    Y_IEX4(:, i) = F_IEX4(h)*Y_IEX4(:, i-1);
    Y_an(:, i) = x_an(tVec(i));
end

%%% -------------------------------- PLOTS -----------------------------
% Stability/Instability domain
fig = figure('Name', 'EX 6 - Stability region', 'NumberTitle', 'off');
re = real(R_RK4);
im = imag(R_RK4);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
fp(1) = plot(reL, imL, '-.', 'LineWidth', 1.5); 
hold on; 
grid on;
fp(3) = fill(reL, imL, 'b', 'FaceAlpha', 0.4);

re = real(R_IEX4);
im = imag(R_IEX4);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
fp(2) = plot(reL, imL, 'LineWidth', 1.5);
fp(4) = fill(reL, imL, 'r', 'FaceAlpha', 0.4);

eigB = eig(B)*0.1;
fp(5) = plot(eigB(1), 0, 'kx', 'markerSize', 10);
fp(6) = plot(eigB(2), 0, 'ks', 'markerSize', 10, 'markerFaceColor', 'g');

xLIM = xlim; xlim([xLIM(1)-0.5 xLIM(2)+0.5]);
yLIM = ylim; ylim([yLIM(1)-0.5 yLIM(2)+0.5]);
axis equal; 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend(fp, {'RK4 Stability margin', 'IEX4 Stability Margin',...
    'RK4 Stability region', 'IEX4 Instability region', ...
    '$h\lambda_1$ of matrix B', '$h\lambda_2$ of matrix B'},...
    'FontSize', 14, 'Location', 'southwest');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
% create zoom to show the eigenvalue
a2 = axes();
a2.Position = [0.15 0.6 0.4 0.25];
box on
re = real(R_IEX4);
im = imag(R_IEX4);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
hold on; 
axis equal; 
grid on;
fill(reL, imL, 'r', 'FaceAlpha', 0.4);
hold on; 
grid on
re = real(R_RK4);
im = imag(R_RK4);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
fill(reL, imL, 'b', 'FaceAlpha', 0.4);
plot(eigB(1), 0, 'kx', 'markerSize', 10);
xL = xlim; 
yL = ylim; 
ax = gca;
ax.XLim = [-0.8 0.8];
ax.YLim = [-0.5 0.5];
%save_fig(fig,'ex6_1');
% RK4 solution
fig =  figure('Name', 'EX 6 - RK4', 'NumberTitle', 'off');
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
fig = figure('Name', 'EX 6 - IEX4', 'NumberTitle', 'off');
plot(t_rk4, Y_IEX4(1, :), 'LineWidth', 1.5); 
grid on; 
hold on;
plot(t_rk4, Y_IEX4(2, :), '--', 'LineWidth', 1.5);
legend({'$x_1$', '$x_2$'}, 'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$x$ [-]', 'FontSize', 18);
title('IEX4 - solution', 'FontSize', 17);
%save_fig(fig,'ex6_4');

% RK4 local integration error
fig = figure('Name', 'EX 6 - RK4 Error', 'NumberTitle', 'off');
semilogy(tVec, abs(Y_an(1, :) - y_rk4(:, 1)'), 'LineWidth', 1.5);
grid on; 
hold on;
semilogy(tVec, abs(Y_an(2, :) - y_rk4(:, 2)'), '--', 'LineWidth', 1.5);
legend({'Error on first component', 'Error on second component'},...
    'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('Error [-]', 'FontSize', 18);
title('Error between analytical and RK4 method', 'FontSize', 17);
%save_fig(fig,'ex6_3');

% IEX4 local integration error
fig = figure('Name', 'EX 6 - IEX4 Error', 'NumberTitle', 'off');
semilogy(tVec, abs(Y_an(1, :) - Y_IEX4(1, :)), 'LineWidth', 1.5); 
grid on; hold on;
semilogy(tVec, abs(Y_an(2, :) - Y_IEX4(2, :)), '-.', 'LineWidth', 1.5);
legend({'Error on first component', 'Error on second component'},...
    'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); 
ylabel('Error [-]', 'FontSize', 18);
title('Error between analytical and IEX4 method', 'FontSize', 17);
%save_fig(fig,'ex6_5');
%system('cd ../Report && pdflatex main.tex > NUL');


%% EX 7
clearvars; close all; clc

% Statement of the problem
x0 = [1 1]';
tlim = [0 3];
h = 0.1;

f = @(t, x) [-5/2*(1 + 8*sin(t))*x(1);... 
              (1 - x(1))*x(2) + x(1)];

% Numerical solution
[t_AB3, Y_AB3, ~]   =  AB3(f, tlim, h, x0);
[t_AM3, Y_AM3, ~]   =  AM3(f, tlim, h, x0);
[t_ABM3, Y_ABM3, ~] = ABM3(f, tlim, h, x0);
[t_BDF3, Y_BDF3, ~] = BDF3(f, tlim, h, x0);

%%% STABILITY DOMAIN --------------------------------------------------------
A = @(alpha) [0 1; -1 2*cos(alpha)];
n = size(x0, 1);

% AB3
F_AB3 = @(h, alpha) [zeros(n, n) eye(n) zeros(n, n); ...
         zeros(n, n) zeros(n, n) eye(n); ...
         5/12*A(alpha)*h -4/3*A(alpha)*h (eye(n) + 23/12*A(alpha)*h)];

% Command mldivide "\" is used instead of inv() since it's more efficient
% and also more robust.

% AM3
F_AM3 = @(h, alpha) max(abs(eig([zeros(n, n) eye(n); ...
         -(eye(n) - 5/12*h*A(alpha))\eye(n)*1/12*h*A(alpha),...
         (eye(n) - 5/12*h*A(alpha))\eye(n)*(eye(n) + 2/3*h*A(alpha))])));
   
% ABM3
F_ABM3 = @(h, alpha) max(abs(eig([zeros(n, n) eye(n) zeros(n, n); ...
         zeros(n, n) zeros(n, n) eye(n); ...
         25/144*(A(alpha)*h)^2, -(1/12*A(alpha)*h + 5/9*(A(alpha)*h)^2),...
         (eye(n) + 13/12*A(alpha)*h + 115/144*(A(alpha)*h)^2)])));

% BDF3
F_BDF3 = @(h, alpha) max(abs(eig([zeros(n, n) eye(n) zeros(n, n); ...
         zeros(n, n) zeros(n, n) eye(n); ...
         eye(n)/(eye(n) - 6/11*A(alpha)*h)*2/11, - (eye(n) -...
         6/11*A(alpha)*h)\eye(n)*9/11, ((eye(n) - 6/11*A(alpha)*h)\eye(n))*18/11])));

% For the stability region code from Ex3 and Ex6 is reused
% AM3 stability region
N = 500;
alphaVec = linspace(pi, 0, N);
R_AM3 = zeros(N, 1);
hGuess = 5;
it = 1;
for alpha = alphaVec
    fun = @(h) max(abs(eig(F_AM3(h, alpha)))) - 1;
    h = fzero(fun, hGuess);
    hGuess = h;
    lambda = eig(A(alpha)); 
    lambda = lambda(1);
    R_AM3(it) = h*lambda;
    it = it + 1;
end

% AB3, ABM3 and BDF3 stability regions
N = 500;
% The alphaVec is not evenly spaced in order to better refine plots in 
% certaina areas
alphaVec = [linspace(0, 80*pi/180, 1/4*N),...
             linspace(80*pi/180, pi/2, 1/2*N),...
                linspace(pi/2, pi, 1/4*N)];

hm_AB3 = 1;
[R_AB3,~] = solveStabilityRegion(F_AB3, alphaVec, A, 1, 0.4);
hm_ABM3 = 2;
[R_ABM3,~] = solveStabilityRegion(F_ABM3, alphaVec, A, 2, 1.5);
hm_BDF3 = 6;
[R_BDF3,~] = solveStabilityRegion(F_BDF3, alphaVec, A, 6, -5);

%%% -------------------------------- PLOTS ------------------------------
% AB3 solution
fig = figure('Name', 'EX 7 - AB3', 'NumberTitle', 'off');
plot(t_AB3, Y_AB3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_AB3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('AB3', 'FontSize', 17);
%legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
%save_fig(fig,'ex7_1');

% AM3 solution
fig = figure('Name', 'EX 7 - AM3', 'NumberTitle', 'off');
plot(t_AB3, Y_AM3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_AM3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('AM3', 'FontSize', 17);
%legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])
%save_fig(fig,'ex7_2');

% ABM3 solution
fig = figure('Name', 'EX 7 - ABM3', 'NumberTitle', 'off');
plot(t_AB3, Y_ABM3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_ABM3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); 
ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('ABM3', 'FontSize', 17);
%legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
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
re = real(R_AB3);
im = imag(R_AB3);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
plot(reL, imL, 'LineWIdth', 1.5); 
hold on; 
grid on;
re = real(R_AM3);
im = imag(R_AM3);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
plot(reL, imL, '-.', 'LineWidth', 1.5);
re = real(R_ABM3);
im = imag(R_ABM3);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
plot(reL, imL, ':', 'LineWidth', 1.5);
re = real(R_BDF3);
im = imag(R_BDF3);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
plot(reL, imL);
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
%   [t, sol, it, fcnE] = newtAn(f, x, invJ, toll)
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
%   [t, sol, it, fcnE] = newtFF(f, x, toll)
%
% INPUT:
%   f        - Function handle [1x1] for evaluating the objective function.
%   x        - Double [nx1] representing the starting guess.
%   toll     - Double [1x1] specifying the tolerance for convergence.
%
% OUTPUT:
%   t        - Double [1x1] representing the computational time needed in seconds.
%   sol      - Double [nx1] representing the solution.
%   it       - Double [1x1] indicating the number of iterations.
%   fcnE     - Double [1x1] representing the number of function evaluations.
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

    delta = sqrt(eps);
    
    if abs(x(1)) > 1
        delta = delta*abs(x(1));
    end
    
    J(:, 1) = ((f([x(1)+delta, x(2)]) - f_x)/delta);
    
    delta = sqrt(eps);

    if abs(x(2)) > 1
        delta = delta*abs(x(2));
    end
    
    J(:, 2) = ((f([x(1), x(2)+delta]) - f_x)/delta);

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
%   [t, sol, it, fcnE] = newtFC(f, x, toll)
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
%   fcnE     - Double [1x1] representing the number of function evaluations.
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

    delta = sqrt(eps);
    
    if abs(x(1)) > 1
        delta = delta*abs(x(1));
    end

    J(:, 1) = ((f([x(1)+delta, x(2)]) - f([x(1)-delta, x(2)]))/(2*delta));

    delta = sqrt(eps);

    if abs(x(2)) > 1
        delta = delta*abs(x(2));
    end

    J(:, 2) = ((f([x(1), x(2)+delta]) - f([x(1), x(2)-delta]))/(2*delta));

    x = x - J\f_x;
    f_x = f(x);
    err = abs(f_x);
    it = it + 1;
    
end
sol = x;


end


function [N, H] = getNH(h, tlim)
% GETNH - Retrieve modified N and H values for ode operators with non-integer step size.
%
%   [N, H] = GETNH(h, tlim)
%
% INPUT:
%   h      - Double [1x1] representing the step size.
%   tlim   - Double [1x2] specifying the time vector limits [initial, final].
%
% OUTPUT:
%   N      - Double [1x1] representing the modified number of points.
%   H      - Double [1x1] representing the modified step size.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   GETNH retrieves modified N (number of points) and H (step size) values
%   for ode operators when the input step size 'h' is not an integer.
%
% -------------------------------------------------------------------------

tVec = tlim(1):h:tlim(2);
N = length(tVec) - 1;
H = tlim(2) - tVec(end);

end


function value = nth_output(N, fcn, varargin)
% NTH_OUTPUT - Retrieve the N-th output from a function with multiple outputs.
%
%   value = NTH_OUTPUT(N, fcn, varargin)
%
% INPUT:
%   N        - Double [1x1] representing the index of the desired output.
%   fcn      - Function handle [1x1] for the function with multiple outputs.
%   varargin - Cell array [1xn] representing inputs required by the function.
%
% OUTPUT:
%   value    - Output of the N-th index from the function fcn.
%
% CREATOR:
%   Cucchi Lorenzo (ID: 10650070)
%
% DESCRIPTION:
%   NTH_OUTPUT is a utility function to retrieve the N-th output from a function
%   that has multiple outputs. This function simplifies the extraction process
%   when calling functions with varargout.
%
% -------------------------------------------------------------------------
[value{1:N}] = fcn(varargin{:});
value = value{N};
end



function [sol] = secantMethod(f, a, b, toll)
% SECANTMETHOD - Retrieve the solution of finding zero using the secant method.
%
%   [t, sol, it, fcnE, cV] = SECANTMETHOD(f, a, b, toll)
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
%   SECANTMETHOD retrieves the solution of finding zero using the secant
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


function [t, Y, ct, fcnE] = RK1(f, tlim, h, x0)
% RK1 - Solve the ODE problem using the first-order Runge-Kutta (RK1) method.
%
%   [t, Y, ct, fcnE] = RK1(f, tlim, h, x0)
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
%   RK1 solves the ordinary differential equation (ODE) problem described by
%   the function f using the first-order Runge-Kutta (RK1) approximation. The
%   function iteratively updates the solution based on the RK1 method until
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
    Y(:, i) = Y(:, i-1) + h*K1;
    
    fcnE = fcnE + 1;
end

Y = Y';
ct = toc(t0);

end

function [t, Y, ct, fcnE] = RK2(f, tlim, h, x0)
% RK2 - Solve the ODE problem using the second-order Runge-Kutta (RK2) method.
%
%   [t, Y, ct, fcnE] = RK2(f, tlim, h, x0)
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
    K2 = f(t(i-1) + h, Y(:, i-1) + K1*h);
    Y(:, i) = Y(:, i-1) + h*(1/2*K1 + 1/2*K2);
    
    fcnE = fcnE + 2;
end

Y = Y';
ct = toc(t0);

end

function [t, Y, ct, fcnE] = RK4(f, tlim, h, x0)
% RK2 - Solve the ODE problem using the second-order Runge-Kutta (RK2) method.
%
%   [t, Y, ct, fcnE] = RK2(f, tlim, h, x0)
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

%%% ---------------------------- EX-4 -----------------------------------



%%% ---------------------------- EX-7 -----------------------------------

function [t, Y, ct] = AB3(f, tlim, h, x0)
% AB3 - Solve the ODE problem using the third-order Adams-Bashforth (AB3) method.
%
%   [t, Y, ct] = AB3(f, tlim, h, x0)
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

t0 = tic;

N = (tlim(2) - tlim(1))/h;
t = linspace(tlim(1), tlim(2), N+1);
Y = [x0'; zeros(N, length(x0))];

% First 3 steps are obtained with RK4 method
for i = 2:3
    K1 = f(t(i-1), Y(i-1, :)');
    K2 = f(t(i-1)+h/2, Y(i-1, :)' + h/2*K1);
    K3 = f(t(i-1)+h/2, Y(i-1, :)' + h/2*K2);
    K4 = f(t(i), Y(i-1, :)' + h*K3);
    Y(i, :) = Y(i-1, :)' + 1/6*h*(K1 + 2*K2 + 2*K3 + K4);
end

for i = 4:N+1
    Y(i, :) = Y(i-1, :)' + h*(23/12*f(t(i-1), Y(i-1, :)')...
               - 16/12*f(t(i-2), Y(i-2, :)') + 5/12*f(t(i-3), Y(i-3, :)'));
end
ct = toc(t0);

end


function [t, Y, ct] = AM3(f, tlim, h, x0)
% AM3 - Solve the ODE problem using the third-order Adams-Moulton (AM3) method.
%
%   [t, Y, ct] = AM3(f, tlim, h, x0)
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

t0 = tic;

N = (tlim(2) - tlim(1))/h;
t = linspace(tlim(1), tlim(2), N+1);
Y = [x0'; zeros(N, length(x0))];

% First 2 steps are obtained with RK4 method
for i = 2:2
    K1 = f(t(i-1), Y(i-1, :)');
    K2 = f(t(i-1)+h/2, Y(i-1, :)' + h/2*K1);
    K3 = f(t(i-1)+h/2, Y(i-1, :)' + h/2*K2);
    K4 = f(t(i), Y(i-1, :)' + h*K3);
    Y(i, :) = Y(i-1, :)' + 1/6*h*(K1 + 2*K2 + 2*K3 + K4);
end

% Anlytical inverse Jacobian for Newton's method
invJ = @(t, Y) [24/(25*h + 200*h*sin(t) + 24), 0; ...
                    -(120*h*(Y(2) - 1))/((5*h*Y(1) - 5*h + 12)...
                    *(25*h + 200*h*sin(t) + 24)), 12/(5*h*Y(1) - 5*h + 12)];

% Function to be zeroed
fun = @(t, Y, Y1, Y2) Y - (Y1 + h*(5/12*[-5/2*(1+8*sin(t))*Y(1);(1-Y(1))*Y(2)+Y(1)] +...
8/12*[-5/2*(1+8*sin(t-h))*Y1(1);(1-Y1(1))*Y1(2)+Y1(1)] - ...
1/12*[-5/2*(1+8*sin(t-(2*h)))*Y2(1);(1-Y2(1))*Y2(2)+Y2(1)]));

toll = 1e-8;

for i = 3:N+1
    err = toll + 1;
    x = Y(i-1, :)';
    fx = fun(t(i), x, Y(i-1, :)', Y(i-2, :)');
    while any(err > toll)
        x = x - invJ(t(i), x)*fx;
        fx = fun(t(i), x, Y(i-1, :)', Y(i-2, :)');
        err = abs(fx);
    end
    Y(i, :) = x;
end
ct = toc(t0);

end


function [t, Y, ct] = ABM3(f, tlim, h, x0)
% ABM3 - Solve the ODE problem using the third-order Adams-Bashforth-Moulton (ABM3) method.
%
%   [t, Y, ct] = ABM3(f, tlim, h, x0)
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

t0 = tic;

N = (tlim(2) - tlim(1))/h;
t = linspace(tlim(1), tlim(2), N+1);
Y = [x0'; zeros(N, length(x0))];

% First 3 steps are obtained with RK4 method
for i = 2:3
    K1 = f(t(i-1), Y(i-1, :)');
    K2 = f(t(i-1)+h/2, Y(i-1, :)' + h/2*K1);
    K3 = f(t(i-1)+h/2, Y(i-1, :)' + h/2*K2);
    K4 = f(t(i), Y(i-1, :)' + h*K3);
    Y(i, :) = Y(i-1, :)' + 1/6*h*(K1 + 2*K2 + 2*K3 + K4);
end


for i = 4:N+1
    xp = Y(i-1, :)' + h*(23/12*f(t(i-1), Y(i-1, :)') - 16/12*f(t(i-2), Y(i-2, :)') + ...
        5/12*f(t(i-3), Y(i-3, :)'));
    Y(i, :) = Y(i-1, :)' + h*(5/12*f(t(i), xp) + 8/12*f(t(i-1), Y(i-1, :)') - ...
        1/12*f(t(i-2), Y(i-2, :)'));
end
ct = toc(t0);

end


function [t, Y, ct] = BDF3(f, tlim, h, x0)
% BDF3 - Solve the ODE problem using the third-order Backward Differentiation Formula (BDF3) method.
%
%   [t, Y, ct] = BDF3(f, tlim, h, x0)
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

t0 = tic;

N = (tlim(2) - tlim(1))/h;
t = linspace(tlim(1), tlim(2), N+1);
Y = [x0'; zeros(N, length(x0))];

% First 3 steps are obtained with RK4 method which is of a higher order
for i = 2:3
    K1 = f(t(i-1), Y(i-1, :)');
    K2 = f(t(i-1)+h/2, Y(i-1, :)' + h/2*K1);
    K3 = f(t(i-1)+h/2, Y(i-1, :)' + h/2*K2);
    K4 = f(t(i), Y(i-1, :)' + h*K3);
    Y(i, :) = Y(i-1, :)' + 1/6*h*(K1 + 2*K2 + 2*K3 + K4);
end
% Function to be zeroed
fun = @(t, Y, Y1, Y2, Y3) Y - (6/11*h*[-5/2*(1+8*sin(t))*Y(1);...
    (1-Y(1))*Y(2)+Y(1)] + 18/11*Y1 - 9/11*Y2 + 2/11*Y3);
% Analytical inverse Jacobian
invJ = @(t, Y) [11/(15*h + 120*h*sin(t) + 11), 0; ...
                -(66*h*(Y(2) - 1))/((6*h*Y(1) - 6*h + 11)...
                *(15*h + 120*h*sin(t) + 11)),...
                11/(6*h*Y(1) - 6*h + 11)];
toll = 1e-8;

for i = 4:N+1
    err = toll + 1;
    x = Y(i-1, :)';
    fx = fun(t(i), x, Y(i-1, :)', Y(i-2, :)', Y(i-3, :)');
    while any(err > toll)
        x = x - invJ(t(i), x)*fx;
        fx = fun(t(i), x, Y(i-1, :)', Y(i-2, :)', Y(i-3, :)');
        err = abs(fx);
    end
    Y(i, :) = x;
end

ct = toc(t0);

end

function [R_F,alphaU] = solveStabilityRegion(F, alphaVec, A, h_max, corr)
% solveStabilityRegion - Compute and analyze the stability region of a given
% linear operator for various parameter values.
%
% PROTOTYPE
%   [R_F, alphaU] = solveStabilityRegion(F, alphaVec, A, h_max, corr)
%
% INPUT:
%   F         - Function handle [1x1] for the linear operator as a function
%               of step size and parameter alpha.
%   alphaVec  - Double [1xM] array specifying the parameter values.
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


hF  = nan(length(alphaVec), 3);
it = 1;
h0Vec = linspace(0, h_max, 3);
%h0Vec = 3;
opt = optimset('TolX', 1e-15);

for alpha = alphaVec
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
R_F = nan(length(alphaVec), 1);
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