% Modeling and Simulation of Aerospace System (2023/2024)
% Assignment # 1
% Author:          Lorenzo Cucchi
% Person code:     10650070
% Matricola code:  

%% Graphics Setup
% Set parameters for good-looking plots
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultLineLineWidth', 1);

%% Ex 1
clearvars; close all; clc

% Function 
func = @(x) [x(2)^2 - x(1) - 2; -x(1)^2 + x(2) + 10];

% Inverse Jacobian of the function
invJ = @(x) (1/(4*x(1)*x(2)-1))*[1, -2*x(2); 2*x(1), -1];

% Tolerance
toll = 1e-8;  

% Finding the first zero z1        
x0 = [1 -4]';
[t11, sol11, n11, fcnE11, c11] = newt2Da(func, x0, invJ, toll);
[t12, sol12, n12, fcnE12, c12] = newt2Df(func, x0, 'forward', toll);
[t13, sol13, n13, fcnE13, c13] = newt2Df(func, x0, 'centered', toll);

% Finding the second zero z2
x0 = [6 5]';
[t21, sol21, n21, fcnE21, c21] = newt2Da(func, x0, invJ, toll);
[t22, sol22, n22, fcnE22, c22] = newt2Df(func, x0, 'forward', toll);
[t23, sol23, n23, fcnE23, c23] = newt2Df(func, x0, 'centered', toll);

%%% ----------------------------- PRINT DATA ------------------------------

fprintf('*************** EXERCISE 1 ***************\n');
fprintf('## Solutions (zero 1): \n');
fprintf('Analytical solution:   [%2.15f, %2.15f]  \n', sol11);
fprintf('Forward differences:   [%2.15f, %2.15f]  \n', sol12);
fprintf('Centered differences:  [%2.15f, %2.15f]  \n\n', sol13);

fprintf('## Solutions (zero 2): \n');
fprintf('Analytical solution:   [%2.15f, %2.15f]  \n', sol21);
fprintf('Forward differences:   [%2.15f, %2.15f]  \n', sol22);
fprintf('Centered differences:  [%2.15f, %2.15f]  \n\n', sol23);

fprintf('## Number of iterations: \n');
fprintf('Analytical solution:   [%d, %d]  \n', n11, n21);
fprintf('Forward differences:   [%d, %d]  \n', n12, n22);
fprintf('Centered differences:  [%d, %d]  \n\n', n13, n23);

fprintf('## Error w.r.t. analytical (zero 1): \n');
fprintf('Analytical solution:   [-, -]  \n');
fprintf('Forward differences:   [%2.15e, %2.15e]  \n', norm(sol11-sol12), norm(sol21-sol22));
fprintf('Centered differences:  [%2.15e, %2.15e]  \n', norm(sol11-sol13), norm(sol21-sol23));


%%% -------------------------------- PLOTS --------------------------------
f1 = @(x1, x2) x2.^2 - x1 - 2;
f2 = @(x1, x2) -x1.^2 + x2 + 10;

% Surface plot of the function f1
fig = figure('Name', 'EX 1 - F1 function', 'NumberTitle', 'off');
[X, Y] = meshgrid(linspace(-5, 5, 500), linspace(-4, 4, 500));
colormap parula
fp(1) = surf(X, Y, f1(X, Y), 'FaceAlpha', 0.7, 'EdgeColor', 'none'); grid on; hold on;
[xRK4, hh] = contour(X, Y, f1(X, Y), [0 0]); delete(hh);
fp(2) = plot3(xRK4(1, 2:end), xRK4(2, 2:end), zeros(size(xRK4(1, 2:end))), '--', 'LineWidth', 3, 'color', 'r');
surf([-5 5; -5 5], [-4 -4; 4 4], zeros(2, 2), 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'k');
ax = gca; ax.FontSize = 15;
xlabel('$x_1$ $[-]$', 'FontSize', 20); ylabel('$x_2$ $[-]$', 'FontSize', 20);
zlabel('$f_x(\mbox{\boldmath$x$})$ $[-]$', 'FontSize', 20);
legend(fp, {'$f_x$', '$f_x = 0$'}, 'FontSize', 18, 'Location', 'best')
save_fig(fig,'ex1_1');

% Surface plot of the function f2
fig = figure('Name', 'EX 1 - F2 function', 'NumberTitle', 'off');
[X, Y] = meshgrid(linspace(-5, 5, 500), linspace(-11, 8, 500));
colormap parula
fp(1) = surf(X, Y, f2(X, Y), 'FaceAlpha', 0.7, 'EdgeColor', 'none'); grid on; hold on;
[xRK4, hh] = contour(X, Y, f2(X, Y), [0 0]); delete(hh);
fp(2) = plot3(xRK4(1, 2:end), xRK4(2, 2:end),zeros(size(xRK4(1, 2:end))), '--', 'LineWidth', 3, 'color', 'r');
surf([-5 5;-5 5], [-11 -11;8 8], zeros(2, 2), 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'k');
ax = gca; ax.FontSize = 15;
xlabel('$x_1$ $[-]$', 'FontSize', 20); 
ylabel('$x_2$ $[-]$', 'FontSize', 20);
zlabel('$f_y(\mbox{\boldmath $x$})$ $[-]$', 'FontSize', 20);
legend(fp, {'$f_y$', '$f_y = 0$'}, 'FontSize', 18, 'Location', 'best')
save_fig(fig,'ex1_2');


% Plot of the solution
fig = figure('Name', 'EX 1 - Solution', 'NumberTitle', 'off');
[X, Y] = meshgrid(linspace(-5, 7, 500), linspace(-8, 8, 500));
contour(X, Y, f1(X, Y), [0 0], 'color', [0 0.45 0.74], 'LineWidth', 1.5); grid on; hold on;
contour(X, Y, f2(X, Y), [0 0], '--', 'color', [0.85 0.33 0.10], 'LineWidth', 1.5);
ax = gca; ax.FontSize = 15;
plot(sol11(1), sol11(2), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'y');
text(sol11(1)+0.2, sol11(2)+0.5, '$\boldmath{z_1}$', 'FontSize', 17);
plot(sol21(1), sol21(2), 'ks', 'MarkerSize', 6, 'MarkerFaceColor', 'y');
text(sol21(1)+0.2, sol21(2)+0.8, '$\boldmath{z_2}$', 'FontSize', 17);
xlabel('$x_1$ $[-]$', 'FontSize', 20); 
ylabel('$x_2$ $[-]$', 'FontSize', 20);
legend({'$f_x = 0$', '$f_y = 0$', 'zeros of $\mbox{\boldmath $f$}(\mbox{\boldmath $x$})$'}, 'FontSize', 15, 'Location', 'best')
save_fig(fig,'ex1_3');



%% EX 2
clearvars; close all; clc

% Statement of the differential problem
x0   = 1; 
tlim = [0 2];
f    = @(t, x) x - 2*t^2 + 2;
uex  = @(t) 2*t.^2 + 4*t - exp(t) + 2;
hvec = [0.5 0.2 0.05 0.01];
N    = length(hvec);

%%% RK2 -------------------------------------------------------------------
% Pre-allocation
Y_RK2         = cell(N, 1);
t_RK2         = cell(N, 1);
globIntErrRK2 = zeros(N, 1);
locIntErrRK2  = cell(N, 1);

for i = 1:N
    % Solution
    [t_RK2{i}, Y_RK2{i}, ~, ~] = RK2(f, tlim, hvec(i), x0);
    % Integration errors
    globIntErrRK2(i) = max(abs(Y_RK2{i}' - uex(t_RK2{i})));
    locIntErrRK2{i} = abs(Y_RK2{i}' - uex(t_RK2{i}));
end

%%% RK4 -------------------------------------------------------------------
% Pre-allocation
Y_RK4         = cell(N, 1);
t_RK4         = cell(N, 1);
globIntErrRK4 = zeros(N, 1);
locIntErrRK4  = cell(N, 1);

for i = 1:N
    % Solution
    [t_RK4{i}, Y_RK4{i}, ~, ~] = RK4(f, tlim, hvec(i), x0);
    % Integration errors
    globIntErrRK4(i) = max(abs(Y_RK4{i}' - uex(t_RK4{i})));
    locIntErrRK4{i} = abs(Y_RK4{i}' - uex(t_RK4{i}));
end

%%% CPU time analysis
Nvec_t = 2.^(0:1:20);
hvec_t = (tlim(2) - tlim(1))./Nvec_t;
N_t = length(hvec_t);

% Preallocation
compTimeRK2   = nan(N_t, 1);
err_fin_RK2_t = nan(N_t, 1);
compTimeRK4   = nan(N_t, 1);
err_fin_RK4_t = nan(N_t, 1);

% Computations
for i = 1:N_t
    [~, Y, compTimeRK2(i), ~] = RK2(f, tlim, hvec_t(i), x0);
    err_fin_RK2_t(i) = abs(uex(tlim(2)) - Y(end));
    [~, Y, compTimeRK4(i), ~] = RK4(f, tlim, hvec_t(i), x0);
    err_fin_RK4_t(i) = abs(uex(tlim(2)) - Y(end));
end

%%% -------------------------------- PLOTS --------------------------------
% RK2 solution
fig = figure('Name', 'EX 2 - solution RK2', 'NumberTitle', 'off');
markerObj = {'square', 'diamond', '^', 'v'};
lineSpecObj = {'--', '-.', ':'};
legendStr = cell(N, 1);
T = linspace(tlim(1), tlim(2), 100);
plot(T, uex(T), 'LineWidth', 2); grid on; hold on;
for i = 1:N
    if i == 1
        plot(t_RK2{i}, Y_RK2{i}, '-s', 'LineWidth', 0.8);
    else
        plot(t_RK2{i}, Y_RK2{i}, lineSpecObj{i-1}, 'LineWidth', 0.8);
    end
    legendStr{i} = strcat('$h=', string(hvec(i)), '$');
end
xlabel('$t$ $[s]$', 'FontSize', 20); ylabel('$x$ $[-]$', 'FontSize', 20);
legendStr{end+1} = '$x_{ex}$';
ax = gca; ax.FontSize = 15;
legend({legendStr{end} legendStr{1:end-1}}, 'FontSize', 14, 'Location', 'best');
save_fig(fig,'ex2_1');

% RK4 solution
fig = figure('Name', 'EX 2 - solution RK4', 'NumberTitle', 'off');
lineSpecObj = {'--', '-.', ':'};
legendStr = cell(N, 1);
T = linspace(tlim(1), tlim(2), 100);
plot(T, uex(T), 'LineWidth', 2); grid on; hold on;
for i = 1:N
    if i == 1
        plot(t_RK4{i}, Y_RK4{i}, '-s', 'LineWidth', 0.8);
    else
        plot(t_RK4{i}, Y_RK4{i}, lineSpecObj{i-1}, 'LineWidth', 0.8);
    end
    legendStr{i} = strcat('$h=', string(hvec(i)), '$');
end
xlabel('$t$ $[s]$', 'FontSize', 20); ylabel('$x$ $[-]$', 'FontSize', 20);
legendStr{end+1} = '$x_{ex}$';
ax = gca; ax.FontSize = 15;
legend({legendStr{end} legendStr{1:end-1}}, 'FontSize', 14, 'Location', 'best');
legendStr = legendStr(1:end-1);
save_fig(fig,'ex2_2');

% Global integration errors of RK2 and RK4
fig = figure('Name', 'EX 2 - global integration error', 'NumberTitle', 'off');
loglog(hvec, globIntErrRK2, '-s'); grid on; hold on;
loglog(hvec, globIntErrRK4, '-^');
loglog(hvec, hvec.^2, '--');
loglog(hvec, 0.09*hvec.^4, '-.');
ax = gca; ax.FontSize = 13;
title('Global integration error', 'FontSize', 17);
xlabel('$h$ $[-]$', 'FontSize', 18); ylabel('$||\mbox{\boldmath $x$}_{ex}-\mbox{\boldmath $x$}||_{\infty}$ $[-]$', 'FontSize', 18);
legend({'RK2', 'RK4', '$h^2$', '$h^4$'}, 'FontSize', 14, 'Location', 'best');
save_fig(fig,'ex2_3');

% Stochastic analysis made upon time
fig = figure('Name', 'EX 2 - ERR vs TIME', 'NumberTitle', 'off');
loglog(err_fin_RK2_t(3:end), compTimeRK2(3:end)*1000, 'o'); hold on; grid on;
loglog(err_fin_RK4_t(3:end), compTimeRK4(3:end)*1000, 's');
ax = gca; ax.FontSize = 13;
title('Error vs Time', 'FontSize', 17);
xlabel('$||\mbox{\boldmath $x$}_{ex}(2)-\mbox{\boldmath $x$}(2)||$ $[-]$', 'FontSize', 18);
ylabel('$CPU-time$ $[ms]$', 'FontSize', 18);
legend({'RK2', 'RK4'}, 'FontSize', 14, 'Location', 'best');
save_fig(fig,'ex2_4');

% Local integration errors of RK2
fig = figure('Name', 'EX 2 - RK2 local integration error', 'NumberTitle', 'off');
for i = 1:N
    if i == N
        semilogy(t_RK2{i}, locIntErrRK2{i}); hold on; grid on;
    else
        semilogy(t_RK2{i}, locIntErrRK2{i}, strcat('-', markerObj{i})); hold on; grid on;
    end
end
ax = gca; ax.FontSize = 13;
title('RK2: local integration error', 'FontSize', 17);
xlabel('$t [s]$', 'FontSize', 18); ylabel('$|x_{RK2}(t)-x(t)|$ $[-]$', 'FontSize', 18);
legend(legendStr, 'FontSize', 14, 'Location', 'best');
save_fig(fig,'ex2_5');

% Local integration errors of RK4
fig = figure('Name', 'EX 2 - RK4 local integration error', 'NumberTitle', 'off');
for i = 1:N
    if i == N
        semilogy(t_RK4{i}, locIntErrRK4{i}); hold on; grid on;
    else
        semilogy(t_RK4{i}, locIntErrRK4{i}, strcat('-', markerObj{i})); hold on; grid on;
    end
end
ax = gca; ax.FontSize = 13;
title('RK4: local integration error', 'FontSize', 17);
xlabel('$t [s]$', 'FontSize', 18); ylabel('$|x_{RK4}(t)-x(t)|$', 'FontSize', 18);
legend(legendStr, 'FontSize', 14, 'Location', 'southeast');
save_fig(fig,'ex2_6');


%% EX 3
clearvars; close all; clc

% Statement of the problem
N     = 1000;
alpha = pi;
toll  = 1e-10;

% RK2 solution
funRK2 = @(h) max(abs(eig(FRK2(h, alpha)))) - 1;
[~, h_RK2, ~, ~, ~] = secantMethod(funRK2, 3, 6, toll);

% RK4 solution
funRK4 = @(h) max(abs(eig(FRK4(h, alpha)))) - 1;
[~, h_RK4, ~, ~, ~] = secantMethod(funRK4, 3, 6, toll);

%%% POINT 3 ---------------------------------------------------------------
alphaVec = [linspace(0, 75*pi/180, 1/10*N), linspace(75*pi/180, pi/2, 8/10*N), linspace(pi/2, pi, 1/10*N)];
% Preallocation
hRK2  = zeros(N, 1);
R_RK2 = zeros(N, 1);

% Solution for RK2
A = @(alpha) [0 1; -1 2*cos(alpha)];
ctr = 1;
for alpha = alphaVec
    fun = @(h) max(abs(eig(FRK2(h, alpha)))) - 1;
    [~, hRK2(ctr), ~, ~, ~] = secantMethod(fun, 3, 6, toll);
    lambda = eig(A(alpha));
    R_RK2(ctr) = hRK2(ctr)*lambda(1);
    ctr = ctr + 1;
end

% Preallocation
hRK4  = nan(N, 3);
R_RK4 = nan(length(alphaVec), 1);
ctr = 1;
h0Vec = linspace(0, 3, 11);
opt = optimset('TolX', 1e-15);

% Solution for RK4 (inner loop to account for multiple zeros of funcion "fun")
for alpha = alphaVec
    fun = @(h) max(abs(eig(FRK4(h, alpha)))) - 1;
    
    sol = nan(length(h0Vec), 1);
    for i = 1:length(h0Vec)
        sol(i) = fzero(fun, h0Vec(i), opt);
    end
    S = nan(1, 3);
    s0 = sort(unique(abs(round(sol, 4))));
    S(1:length(s0)) = s0;
    hRK4(ctr, 1:length(S)) = S;
    
    ctr = ctr + 1;
end

% Collect all the h founded
ctr = 1;
hRK4_sorted = zeros(length(alphaVec), 1);
alphaU = zeros(length(alphaVec), 1);
for i = 1:length(alphaVec)
    lambda = eig(A(alphaVec(i))); lambda = lambda(1);
    if sum(isnan(hRK4(i, :))) == 1
        hRK4_sorted(ctr) = hRK4(i, 2);
        alphaU(ctr) = alphaVec(i);
        R_RK4(ctr) = hRK4_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hRK4(i, :))) == 2
        hRK4_sorted(ctr) = hRK4(i, 1);
        alphaU(ctr) = alphaVec(i);
        R_RK4(ctr) = hRK4_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hRK4(i, :))) == 0
        hRK4_sorted(ctr) = hRK4(i, 2);
        alphaU(ctr) = alphaVec(i);
        R_RK4(ctr) = hRK4_sorted(ctr)*lambda;
        ctr = ctr + 1;
        hRK4_sorted(ctr) = hRK4(i, 3);
        alphaU(ctr) = alphaVec(i);
        R_RK4(ctr) = hRK4_sorted(ctr)*lambda;
        ctr = ctr + 1;
    end
end

% Sort the values
R_RK4(end+1)  = 0;
alphaU(end+1) = pi/2;

A = zeros(size(R_RK4));
B = zeros(size(R_RK4));

for i = 1:length(R_RK4)
    A(i) = atan2(imag(R_RK4(i)), real(R_RK4(i)) + 2);
    B(i) = atan2(imag(R_RK4(i)), real(R_RK4(i)));
end
[~, ind] = sort(A);
R_RK4 = R_RK4(ind);
alphaU = alphaU(ind);

%%% FROM EXERCISE 2 -------------------------------------------------------
f = @(t) 2*t^2+4*t-exp(t)+2;
hvec = [0.5 0.2 0.05 0.01];
N = length(hvec);

%%% -------------------------------- PLOTS --------------------------------
% h solutions
fig = figure('Name', 'EX 3 - h', 'NumberTitle', 'off');
index = find(alpha <= 0);
plot(alphaVec*180/pi, hRK2, '-.', 'LineWidth', 1.5); grid on; hold on;
plot(alphaU*180/pi, abs(R_RK4), 'LineWidth', 1.5);
xlabel('$\alpha$ $[deg]$', 'FontSize', 18); ylabel('$h$ $[-]$', 'FontSize', 18);
legend({'RK2', 'RK4'}, 'FontSize', 14, 'Location', 'best');
title('Solution of the statement', 'FontSize', 17);
save_fig(fig,'ex3_1');
% Stable domains
fig = figure('Name', 'EX 3 - Regions', 'NumberTitle', 'off');
re = real(R_RK2);
im = imag(R_RK2);
fp(1) = plot(re, im, '-.', 'color', [0 0.45 0.74], 'LineWidth', 1.5); hold on; grid on;
im = -im;
plot(re, im, '-.', 'color', [0 0.45 0.74], 'LineWidth', 1.5);
re = real(R_RK4);
im = imag(R_RK4);
fp(2) = plot(re, im, 'color', [0.85 0.33 0.10], 'LineWidth', 1.5); hold on; grid on;
im = -im;
plot(re, im, 'color', [0.85 0.33 0.10], 'LineWidth', 1.5);
axis equal; ax = gca; addAxes(ax);
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
legend(fp, {'RK2 Stability margin',...
    'RK4 Stability margin'},...
    'FontSize', 11, 'Location', 'best');
title('Stability regions', 'FontSize', 17);
% a2 = axes(); % box of figure 1
% a2.Position = [0.53 0.63 0.3 0.25];
% box on
% hold on; axis equal; grid on;
% re = real(R_RK2);
% im = imag(R_RK2);
% plot(re, im, '-.', 'color', [0 0.45 0.74]);
% plot(re, -im, '-.', 'color', [0 0.45 0.74]);
% re = real(R_RK4);
% im = imag(R_RK4);
% plot(re, im, 'color', [0.85 0.33 0.10]);
% plot(re, -im, 'color', [0.85 0.33 0.10]);
% xlim([-0.1 0.1])
save_fig(fig,'ex3_2');

%%% PRINT DATA ------------------------------------------------------------
fprintf('*************** EXERCISE 3 ***************\n');
fprintf('## Solutions (h): \n');
fprintf('RK2 method:   %2.15e  \n', h_RK2);
fprintf('RK4 method:   %2.15e  \n\n', h_RK4);



%% EX 4
clearvars; close all; clc

% Statement of the problem
x0 = [1 1]';
A = @(alpha) [0 1; -1 2*cos(alpha)];
tlim = [0 1];

% Analytical solution
x_analytical = @(t, alpha) expm(t*A(alpha))*x0;

%%% NEEDED FUNCTIONS ------------------------------------------------------
% Useful
getN = @(h) nth_output(1, @getNH, h, tlim);
getHbar = @(h) nth_output(2, @getNH, h, tlim);

% Linear operator of the methods
F_RK1 = @(h, alpha) eye(length(x0)) + h*A(alpha);
F_RK2 = @(h, alpha) eye(length(x0)) + h*A(alpha) + (h^2)/2*A(alpha)^2;
F_RK4 = @(h, alpha) eye(length(x0)) + h*A(alpha) + (h^2)/2*A(alpha)^2 + ((h^3)/6)*A(alpha)^3 + ((h^4)/24)*A(alpha)^4;

% Solution functions with the three methods
x_RK1 = @(h, alpha) F_RK1(getHbar(h), alpha)*(F_RK1(h, alpha)^getN(h))*x0;
x_RK2 = @(h, alpha) F_RK2(getHbar(h), alpha)*(F_RK2(h, alpha)^getN(h))*x0;
x_RK4 = @(h, alpha) F_RK4(getHbar(h), alpha)*(F_RK4(h, alpha)^getN(h))*x0;

% Functions to be solved
fun1 = @(h, alpha, toll) norm(x_analytical(tlim(2), alpha) - x_RK1(h, alpha), 'inf') - toll;
fun2 = @(h, alpha, toll) norm(x_analytical(tlim(2), alpha) - x_RK2(h, alpha), 'inf') - toll;
fun4 = @(h, alpha, toll) norm(x_analytical(tlim(2), alpha) - x_RK4(h, alpha), 'inf') - toll;

%%% CALCULATIONS ----------------------------------------------------------
% Pre-allocation
alphaVec = linspace(0, pi, 100);
tolVec = [1e-3, 1e-4, 1e-5, 1e-6];

h_RK1 = cell(length(tolVec), 1);
h_RK2 = cell(length(tolVec), 1);
h_RK4 = cell(length(tolVec), 1);
R_RK1 = cell(length(tolVec), 1);
R_RK2 = cell(length(tolVec), 1);
R_RK4 = cell(length(tolVec), 1);
fcnE1 = zeros(length(tolVec), 1);
fcnE2 = zeros(length(tolVec), 1);
fcnE4 = zeros(length(tolVec), 1);
opt = optimoptions('fsolve', 'Display', 'none');

% Loop to find all the h values that zeroed the functions fun(i)
for i = 1:length(tolVec)
    toll = tolVec(i);
    h_RK1{i} = zeros(length(alphaVec), 1);
    h_RK2{i} = zeros(length(alphaVec), 1);
    h_RK4{i} = zeros(length(alphaVec), 1);
    R_RK1{i} = zeros(length(alphaVec), 1);
    R_RK2{i} = zeros(length(alphaVec), 1);
    R_RK4{i} = zeros(length(alphaVec), 1);
    F11 = @(h, alpha) fun1(h, alpha, toll);
    F12 = @(h, alpha) fun2(h, alpha, toll);
    F14 = @(h, alpha) fun4(h, alpha, toll);
    
    for j = 1:length(alphaVec)
        F21 = @(h) F11(h, alphaVec(j));
        F22 = @(h) F12(h, alphaVec(j));
        F24 = @(h) F14(h, alphaVec(j));
        
        h_RK1{i}(j) = fsolve(F21, 5e-5, opt);
        h_RK2{i}(j) = fzero(F22, [5e-6, 1]);
        h_RK4{i}(j) = fzero(F24, [5e-6, 1]);
        
        lambda = eig(A(alphaVec(j))); lambda = lambda(1);
        
        R_RK1{i}(j) = h_RK1{i}(j)*lambda;
        R_RK2{i}(j) = h_RK2{i}(j)*lambda;
        R_RK4{i}(j) = h_RK4{i}(j)*lambda;
    end
    
end

% Obtain function evaluations number
for i = 1:length(tolVec)
    toll = tolVec(i);
    [~, ~, ~, fcnE1(i)] = RK1(@(t, x) A(pi)*x, tlim, h_RK1{i}(end), x0);
    [~, ~, ~, fcnE2(i)] = RK2(@(t, x) A(pi)*x, tlim, h_RK2{i}(end), x0);
    [~, ~, ~, fcnE4(i)] = RK4(@(t, x) A(pi)*x, tlim, h_RK4{i}(end), x0);
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
axis equal; ax = gca; addAxes(ax);
legend(fp, {'Tol = $10^{-3}$', 'Tol = $10^{-4}$', 'Tol = $10^{-5}$', 'Tol = $10^{-6}$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
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
xlim([-2e-5 3e-5]); ax = gca; addAxes(ax);
save_fig(fig,'ex4_1');
% RK" solutions
fig = figure('Name', 'EX 4 - RK2', 'NumberTitle', 'off');
lineSpecObj = {'-.', '--', ':'};
for i = 1:length(tolVec)
    re = real(R_RK2{i});
    im = imag(R_RK2{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    if i == 1
        fp(i) = plot(reL, imL, 'LineWidth', 1.5); hold on; grid on;
    else
        fp(i) = plot(reL, imL, lineSpecObj{i-1}, 'LineWidth', 1.5);
    end
end
axis equal; ax = gca; addAxes(ax);
legend(fp, {'Tol = $10^{-3}$', 'Tol = $10^{-4}$', 'Tol = $10^{-5}$', 'Tol = $10^{-6}$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
title('RK2', 'FontSize', 17);
save_fig(fig,'ex4_2');
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
axis equal; ax = gca; addAxes(ax);
legend(fp, {'Tol = $10^{-3}$', 'Tol = $10^{-4}$', 'Tol = $10^{-5}$', 'Tol = $10^{-6}$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
title('RK4', 'FontSize', 17);
save_fig(fig,'ex4_3');
% Function evaluations vs tolerance plot
fig = figure('Name', 'EX 4 - f evaluations', 'NumberTitle', 'off');
loglog(tolVec, fcnE1, '-s', 'LineWidth', 1.5); hold on;
loglog(tolVec, fcnE2, '-^', 'LineWidth', 1.5)
loglog(tolVec, fcnE4, '-o', 'LineWidth', 1.5); grid on;
title('f evaluations vs Tolerance', 'FontSize', 17);
xlabel('tol [-]', 'FontSize', 18); ylabel('f evaluations [-]', 'FontSize', 18);
legend({'RK1', 'RK2', 'RK4'}, 'FontSize', 14, 'Location', 'best');
save_fig(fig,'ex4_4');
%% EX 4
clearvars; close all; clc; warning off;

% Statement of the problem
A = @(alpha) [0 1; -1 2*cos(alpha)];
alphaVec = linspace(0, pi, 500);
thetaVec = [0.4 0.1 0.3 0.7 0.9];

% Linear operator
F_BItheta = @(h, alpha, theta) (eye(2) - (1 - theta)*h*A(alpha) + ...
    ((((1 - theta)*h)^2)/2)*A(alpha)^2)\...
    (eye(2) + theta*h*A(alpha) + (((theta*h)^2)/2)*A(alpha)^2);

% Pre-allocation
h = cell(length(thetaVec), 1);
R = cell(length(thetaVec), 1);

% Calculations
for i = 1:length(thetaVec)
    h{i} = zeros(length(alphaVec), 1);
    R{i} = zeros(length(alphaVec), 1);
    
    for j = 1:length(alphaVec)
        fun = @(h) max(abs(eig(F_BItheta(h, alphaVec(j), thetaVec(i))))) - 1;
        try
            h{i}(j) = fzero(fun, 9);
            lambda = eig(A(alphaVec(j))); lambda = lambda(1);
            R{i}(j) = h{i}(j)*lambda;
        catch me
        end
    end
end
warning on;


%%% -------------------------------- PLOTS --------------------------------
% BI2_theta unstable domains
figure('Name', 'EX 5 - BI2', 'NumberTitle', 'off');
lineSpecObj = {'-.', '--', ':', '-'};
str = {''};
for i = 1:length(thetaVec)
    re = real(R{i});
    im = imag(R{i});
    reL = [re; re(end:-1:1)];
    imL = [im; -im(end:-1:1)];
    if i == 1
        fp(i) = plot(reL, imL, 'LineWidth', 1.5); hold on; grid on;
    elseif i == length(thetaVec)
        fp(i) = plot(reL, imL, lineSpecObj{i-1}, 'LineWidth', 0.1);
    else
        fp(i) = plot(reL, imL, lineSpecObj{i-1}, 'LineWidth', 1.5);
    end
    str{i} = strcat('$\theta$', " = ", '$', string(thetaVec(i)),'$');
end
xLIM = xlim; xlim([xLIM(1)-0.5 xLIM(2)+0.5]);
axis equal; ax = gca; addAxes(ax);
legend(fp, str, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
title('Unstable regions of $BI2_{\theta}$', 'FontSize', 17);

% h solutions
figure('Name', 'EX 5 - BI2 h', 'NumberTitle', 'off');
for i = 1:length(thetaVec)
    re = real(R{i});
    im = imag(R{i});
    if i == 1
        alpha = atan2d(im, re);
        [alpha, IA, IC] = unique(alpha);
        R{i} = R{i}(IA);
        ind = find(alpha <= 90, 1, 'last');
        R{i} = [R{i}(1:ind); 0; R{i}(ind+1:end)];
        alpha = [alpha(1:ind); 90; alpha(ind+1:end)];
        fp(i) = plot(alpha, abs(R{i}), 'LineWidth', 1.5); hold on; grid on;
    elseif i > 1 && i < length(thetaVec)
        alpha = atan2d(im, re);
        [alpha, IA, IC] = unique(alpha);
        R{i} = R{i}(IA);
        ind = find(alpha <= 90, 1, 'last');
        R{i} = [R{i}(1:ind); 0; R{i}(ind+1:end)];
        alpha = [alpha(1:ind); 90; alpha(ind+1:end)];
        fp(i) = plot(alpha, abs(R{i}), lineSpecObj{i-1}, 'LineWidth', 1.5);
    else
        alpha = atan2d(im, re);
        [alpha, IA, IC] = unique(alpha);
        R{i} = R{i}(IA);
        ind = find(alpha <= 90, 1, 'last');
        R{i} = [R{i}(1:ind); 0; R{i}(ind+1:end)];
        alpha = [alpha(1:ind); 90; alpha(ind+1:end)];
        fp(i) = plot(alpha, abs(R{i}), lineSpecObj{i-1}, 'LineWidth', 0.1);
    end
end
xlim([0 180]);
legend(fp, str, 'FontSize', 14, 'Location', 'best');
xlabel('$\alpha$ [deg]', 'FontSize', 18); ylabel('$h$ [-]', 'FontSize', 18);
title('Minimum $h$ for stability', 'FontSize', 17);



%% EX 5
clearvars; close all; clc

% Statement of the problem
B = [-180.5 219.5; 179.5 -220.5];
x0 = [1 1]';
tlim = [0 5];

% Analytical solution
x_analytical = @(t) expm(t*B)*x0;

f = @(t, x) B*x;

% RK4 solution
[t_RK4, Y_RK4, ~, ~] = RK4(f, tlim, 0.1, x0);

%%% RK4 stability domain --------------------------------------------------
N = 1000;
A = @(alpha) [0 1; -1 2*cos(alpha)];
alphaVec = [linspace(0, 75*pi/180, 1/10*N), linspace(75*pi/180, pi/2, 8/10*N), linspace(pi/2, pi, 1/10*N)];

% Pre-allocation
hRK4  = nan(N, 3);
R_RK4 = nan(length(alphaVec), 1);
ctr = 1;
h0Vec = linspace(0, 3, 11);
opt = optimset('TolX', 1e-15);
for alpha = alphaVec
    fun = @(h) max(abs(eig(FRK4(h, alpha)))) - 1;
    
    sol = nan(length(h0Vec), 1);
    for i = 1:length(h0Vec)
        sol(i) = fzero(fun, h0Vec(i), opt);
    end
    S = nan(1, 3);
    s0 = sort(unique(abs(round(sol, 4))));
    S(1:length(s0)) = s0;
    hRK4(ctr, 1:length(S)) = S;
    
    ctr = ctr + 1;
end

ctr = 1;
hRK4_sorted = zeros(length(alphaVec), 1);
alphaU = zeros(length(alphaVec), 1);
for i = 1:length(alphaVec)
    lambda = eig(A(alphaVec(i))); lambda = lambda(1);
    if sum(isnan(hRK4(i, :))) == 1
        hRK4_sorted(ctr) = hRK4(i, 2);
        alphaU(ctr) = alphaVec(i);
        R_RK4(ctr) = hRK4_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hRK4(i, :))) == 2
        hRK4_sorted(ctr) = hRK4(i, 1);
        alphaU(ctr) = alphaVec(i);
        R_RK4(ctr) = hRK4_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hRK4(i, :))) == 0
        hRK4_sorted(ctr) = hRK4(i, 2);
        alphaU(ctr) = alphaVec(i);
        R_RK4(ctr) = hRK4_sorted(ctr)*lambda;
        ctr = ctr + 1;
        hRK4_sorted(ctr) = hRK4(i, 3);
        alphaU(ctr) = alphaVec(i);
        R_RK4(ctr) = hRK4_sorted(ctr)*lambda;
        ctr = ctr + 1;
    end
end

% Sort obtained values
R_RK4(end+1)  = 0;
A = zeros(size(R_RK4));
B = zeros(size(R_RK4));
for i = 1:length(R_RK4)
    A(i) = atan2(imag(R_RK4(i)), real(R_RK4(i)) + 2);
    B(i) = atan2(imag(R_RK4(i)), real(R_RK4(i)));
end
[~, ind] = sort(A);
R_RK4 = R_RK4(ind);

% IEX4 instability domain -------------------------------------------------
alphaVec = linspace(0, pi, 500);
hIEX4  = zeros(length(alphaVec), 1);
R_IEX4 = zeros(length(alphaVec), 1);
A = @(alpha) [0 1; -1 2*cos(alpha)];

FIEX4 = @(h, alpha) (-1/6)*((eye(2)-h*A(alpha)))^(-1)+4*(((eye(2)-(h/2)*A(alpha))^2))^(-1)-...
    (27/2)*(((eye(2)-(h/3)*A(alpha))^3))^(-1)+(32/3)*(((eye(2)-(h/4)*A(alpha))^4))^(-1); 

hGuess = 10;
ctr = 1;
for alpha = alphaVec
    fun = @(h) max(abs(eig(FIEX4(h, alpha)))) - 1;
    hIEX4(ctr) = fzero(fun, hGuess);
    hGuess = hIEX4(ctr);
    lambda = eig(A(alpha)); lambda = lambda(1);
    R_IEX4(ctr) = hIEX4(ctr)*lambda;
    ctr = ctr + 1;
end

% Solution with IEX4 ------------------------------------------------------
B = [-180.5 219.5; 179.5 -220.5];
h = 0.1;
tVec = tlim(1):h:tlim(2);
Y_IEX4 = [x0, zeros(2, length(tVec)-1)];
F_IEX4 = @(h) (-1/6)*((eye(2)-h*B))^(-1)+4*(((eye(2)-(h/2)*B)^2))^(-1)-...
    (27/2)*(((eye(2)-(h/3)*B)^3))^(-1)+(32/3)*(((eye(2)-(h/4)*B)^4))^(-1); 

for i = 2:length(tVec)
    Y_IEX4(:, i) = F_IEX4(h)*Y_IEX4(:, i-1);
end

% Analytical solution
Y_analytical = [x0, zeros(2, length(tVec)-1)];
for i = 2:length(tVec)
    Y_analytical(:, i) = x_analytical(tVec(i));
end

%%% -------------------------------- PLOTS --------------------------------
% Stability/Instability domain
figure('Name', 'EX 7 - Stability region', 'NumberTitle', 'off');
re = real(R_RK4);
im = imag(R_RK4);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
fp(1) = plot(reL, imL, '-.', 'LineWidth', 1.5); hold on; grid on;
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
axis equal; ax = gca; addAxes(ax);
legend(fp, {'RK4 Stability margin', 'IEX4 Stability Margin',...
    'RK4 Stability region', 'IEX4 Instability region', ...
    '$h\lambda_1$ of matrix B', '$h\lambda_2$ of matrix B'}, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); ylabel('$Im\{h\lambda\}$', 'FontSize', 18);

a2 = axes();
a2.Position = [0.15 0.6 0.4 0.25];
box on
re = real(R_IEX4);
im = imag(R_IEX4);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
hold on; axis equal; grid on;
fill(reL, imL, 'r', 'FaceAlpha', 0.4);
hold on; grid on
re = real(R_RK4);
im = imag(R_RK4);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
fill(reL, imL, 'b', 'FaceAlpha', 0.4);
plot(eigB(1), 0, 'kx', 'markerSize', 10);
xL = xlim; yL = ylim; %xline(0); yline(0);
ax = gca;
ax.XLim = [-0.8 0.8];
ax.YLim = [-0.5 0.5];

% RK4 solution
figure('Name', 'EX 7 - RK4', 'NumberTitle', 'off');
plot(t_RK4, Y_RK4(:, 1), 'LineWidth', 1.5); grid on; hold on;
plot(t_RK4, Y_RK4(:, 2), '--', 'LineWidth', 1.5);
legend({'$x_1$', '$x_2$'}, 'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); ylabel('$x$ [-]', 'FontSize', 18);
title('RK4 - solution', 'FontSize', 17);

% IEX4 solution
figure('Name', 'EX 7 - IEX4', 'NumberTitle', 'off');
plot(t_RK4, Y_IEX4(1, :), 'LineWidth', 1.5); grid on; hold on;
plot(t_RK4, Y_IEX4(2, :), '--', 'LineWidth', 1.5);
legend({'$x_1$', '$x_2$'}, 'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); ylabel('$x$ [-]', 'FontSize', 18);
title('IEX4 - solution', 'FontSize', 17);

% RK4 local integration error
figure('Name', 'EX 7 - Error RK4', 'NumberTitle', 'off');
semilogy(tVec, abs(Y_analytical(1, :) - Y_RK4(:, 1)'), 'LineWidth', 1.5); grid on; hold on;
semilogy(tVec, abs(Y_analytical(2, :) - Y_RK4(:, 2)'), '--', 'LineWidth', 1.5);
legend({'Error on first component', 'Error on second component'}, 'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); ylabel('Error [-]', 'FontSize', 18);
title('Error between analytical and RK4 method', 'FontSize', 17);

% IEX4 local integration error
figure('Name', 'EX 7 - Error IEX4', 'NumberTitle', 'off');
semilogy(tVec, abs(Y_analytical(1, :) - Y_IEX4(1, :)), 'LineWidth', 1.5); grid on; hold on;
semilogy(tVec, abs(Y_analytical(2, :) - Y_IEX4(2, :)), '-.', 'LineWidth', 1.5);
legend({'Error on first component', 'Error on second component'}, 'FontSize', 14, 'Location', 'best');
xlabel('Time [s]', 'FontSize', 18); ylabel('Error [-]', 'FontSize', 18);
title('Error between analytical and IEX4 method', 'FontSize', 17);



%% EX 6
clearvars; close all; clc

% Statement of the problem
x0 = [1 1]';
tlim = [0 3];
h = 0.1;

f = @(t, x) [-5/2*(1 + 8*sin(t))*x(1); (1 - x(1))*x(2)+x(1)];

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

% AM3
F_AM3 = @(h, alpha) max(abs(eig([zeros(n, n) eye(n); ...
         -inv(eye(n) - 5/12*h*A(alpha))*1/12*h*A(alpha), inv(eye(n) - 5/12*h*A(alpha))*(eye(n) + 2/3*h*A(alpha))])));

     
% ABM3
F_ABM3 = @(h, alpha) max(abs(eig([zeros(n, n) eye(n) zeros(n, n); ...
         zeros(n, n) zeros(n, n) eye(n); ...
         25/144*(A(alpha)*h)^2, -(1/12*A(alpha)*h + 5/9*(A(alpha)*h)^2), (eye(n) + 13/12*A(alpha)*h + 115/144*(A(alpha)*h)^2)])));

% BDF3
F_BDF3 = @(h, alpha) max(abs(eig([zeros(n, n) eye(n) zeros(n, n); ...
         zeros(n, n) zeros(n, n) eye(n); ...
         inv(eye(n) - 6/11*A(alpha)*h)*2/11, - inv(eye(n) - 6/11*A(alpha)*h)*9/11, inv(eye(n) - 6/11*A(alpha)*h)*18/11])));

% AM3 stability region
N = 100;
alphaVec = linspace(pi, 0, N);
R_AM3 = zeros(N, 1);
hGuess = 5;
ctr = 1;
for alpha = alphaVec
    fun = @(h) max(abs(eig(F_AM3(h, alpha)))) - 1;
    h = fzero(fun, hGuess);
    hGuess = h;
    lambda = eig(A(alpha)); lambda = lambda(1);
    R_AM3(ctr) = h*lambda;
    ctr = ctr + 1;
end

% AB3, ABM3 and BDF3 stability regions
N = 500;
alphaVec = [linspace(0, 80*pi/180, 1/5*N), linspace(80*pi/180, pi/2, 3/5*N), linspace(pi/2, pi, 1/5*N)];
hAB3   = nan(N, 3);
hABM3  = nan(N, 3);
hBDF3  = nan(N, 3);
R_AB3  = nan(length(alphaVec), 1);
R_ABM3 = nan(length(alphaVec), 1);
R_BDF3 = nan(length(alphaVec), 1);

h0VecAB3  = linspace(0, 1, 11);
h0VecABM3 = linspace(0, 2, 11);
h0VecBDF3 = linspace(0, 6, 11);

opt = optimset('TolX', 1e-15);
ctr = 1;
for alpha = alphaVec
    funAB3  = @(h) max(abs(eig(F_AB3(h, alpha)))) - 1;
    funABM3 = @(h) max(abs(eig(F_ABM3(h, alpha)))) - 1;
    funBDF3 = @(h) max(abs(eig(F_BDF3(h, alpha)))) - 1;
    
    solAB3  = nan(length(h0VecAB3), 1);
    solABM3 = nan(length(h0VecABM3), 1);
    solBDF3 = nan(length(h0VecBDF3), 1);
    
    for i = 1:length(h0VecAB3)
        solAB3(i)  = fzero(funAB3, h0VecAB3(i), opt);
        solABM3(i) = fzero(funABM3, h0VecABM3(i), opt);
        solBDF3(i) = fzero(funBDF3, h0VecBDF3(i), opt);
    end
    SAB3  = nan(1, 3);
    SABM3 = nan(1, 3);
    SBDF3 = nan(1, 3);
    
    s0AB3 = sort(unique(abs(round(solAB3, 4))));
    SAB3(1:length(s0AB3)) = s0AB3;
    hAB3(ctr, 1:length(SAB3)) = SAB3;
    
    s0ABM3 = sort(unique(abs(round(solABM3, 4))));
    SABM3(1:length(s0ABM3)) = s0ABM3;
    hABM3(ctr, 1:length(SABM3)) = SABM3;
    
    s0BDF3 = sort(unique(abs(round(solBDF3, 4))));
    SBDF3(1:length(s0BDF3)) = s0BDF3;
    hBDF3(ctr, 1:length(SBDF3)) = SBDF3;
    
    ctr = ctr + 1;
end

ctr = 1;
hAB3_sorted  = zeros(length(alphaVec), 1);
hABM3_sorted = zeros(length(alphaVec), 1);
hBDF3_sorted = zeros(length(alphaVec), 1);

alphaUAB3 = zeros(length(alphaVec), 1);
alphaUABM3 = zeros(length(alphaVec), 1);
alphaUBDF3 = zeros(length(alphaVec), 1);

for i = 1:length(alphaVec)
    lambda = eig(A(alphaVec(i))); lambda = lambda(1);
    % AB3
    if sum(isnan(hAB3(i, :))) == 1
        hAB3_sorted(ctr) = hAB3(i, 2);
        alphaUAB3(ctr) = alphaVec(i);
        R_AB3(ctr) = hAB3_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hAB3(i, :))) == 2
        hAB3_sorted(ctr) = hAB3(i, 1);
        alphaUAB3(ctr) = alphaVec(i);
        R_AB3(ctr) = hAB3_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hAB3(i, :))) == 0
        hAB3_sorted(ctr) = hAB3(i, 2);
        alphaUAB3(ctr) = alphaVec(i);
        R_AB3(ctr) = hAB3_sorted(ctr)*lambda;
        ctr = ctr + 1;
        hAB3_sorted(ctr) = hAB3(i, 3);
        alphaUAB3(ctr) = alphaVec(i);
        R_AB3(ctr) = hAB3_sorted(ctr)*lambda;
        ctr = ctr + 1;
    end
    % ABM3
    lambda = eig(A(alphaVec(i))); lambda = lambda(1);
    if sum(isnan(hABM3(i, :))) == 1
        hABM3_sorted(ctr) = hABM3(i, 2);
        alphaUABM3(ctr) = alphaVec(i);
        R_ABM3(ctr) = hABM3_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hABM3(i, :))) == 2
        hABM3_sorted(ctr) = hABM3(i, 1);
        alphaUABM3(ctr) = alphaVec(i);
        R_ABM3(ctr) = hABM3_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hABM3(i, :))) == 0
        hABM3_sorted(ctr) = hABM3(i, 2);
        alphaUABM3(ctr) = alphaVec(i);
        R_ABM3(ctr) = hABM3_sorted(ctr)*lambda;
        ctr = ctr + 1;
        hABM3_sorted(ctr) = hABM3(i, 3);
        alphaUABM3(ctr) = alphaVec(i);
        R_ABM3(ctr) = hABM3_sorted(ctr)*lambda;
        ctr = ctr + 1;
    end
    % BDF3
    lambda = eig(A(alphaVec(i))); lambda = lambda(1);
    if sum(isnan(hBDF3(i, :))) == 1
        hBDF3_sorted(ctr) = hBDF3(i, 2);
        alphaUBDF3(ctr) = alphaVec(i);
        R_BDF3(ctr) = hBDF3_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hBDF3(i, :))) == 2
        hBDF3_sorted(ctr) = hBDF3(i, 1);
        alphaUBDF3(ctr) = alphaVec(i);
        R_BDF3(ctr) = hBDF3_sorted(ctr)*lambda;
        ctr = ctr + 1;
    elseif sum(isnan(hBDF3(i, :))) == 0
        hBDF3_sorted(ctr) = hBDF3(i, 2);
        alphaUBDF3(ctr) = alphaVec(i);
        R_BDF3(ctr) = hBDF3_sorted(ctr)*lambda;
        ctr = ctr + 1;
        hBDF3_sorted(ctr) = hBDF3(i, 3);
        alphaUBDF3(ctr) = alphaVec(i);
        R_BDF3(ctr) = hBDF3_sorted(ctr)*lambda;
        ctr = ctr + 1;
    end
end
R_AB3(end+1)   = 0;
R_ABM3(end+1)  = 0;
R_BDF3 = [R_BDF3(1); R_BDF3(end:-1:1)];

AAB3  = zeros(size(R_AB3));
AABM3 = zeros(size(R_ABM3));
ABDF3 = zeros(size(R_BDF3));

BAB3  = zeros(size(R_AB3));
BABM3 = zeros(size(R_ABM3));
BBDF3 = zeros(size(R_BDF3));

for i = 1:length(R_AB3)
    AAB3(i)  = atan2(imag(R_AB3(i)), real(R_AB3(i)) + 0.4);
    BAB3(i)  = atan2(imag(R_AB3(i)), real(R_AB3(i)));
    
    AABM3(i) = atan2(imag(R_ABM3(i)), real(R_ABM3(i)) + 1.5);
    BABM3(i) = atan2(imag(R_ABM3(i)), real(R_ABM3(i)));
    
    ABDF3(i) = atan2(imag(R_BDF3(i)), real(R_BDF3(i)) -5);
    BBDF3(i) = atan2(imag(R_BDF3(i)), real(R_BDF3(i)));
end
[~, ind] = sort(AAB3);
R_AB3 = R_AB3(ind);
[~, ind] = sort(AABM3);
R_ABM3 = R_ABM3(ind);
[~, ind] = sort(ABDF3);
R_BDF3 = R_BDF3(ind);

%%% -------------------------------- PLOTS --------------------------------
% AB3 solution
figure('Name', 'EX 7 - AB3', 'NumberTitle', 'off');
plot(t_AB3, Y_AB3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_AB3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('AB3', 'FontSize', 17);
%legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])

% AM3 solution
figure('Name', 'EX 7 - AM3', 'NumberTitle', 'off');
plot(t_AB3, Y_AM3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_AM3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('AM3', 'FontSize', 17);
%legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])

% ABM3 solution
figure('Name', 'EX 7 - ABM3', 'NumberTitle', 'off');
plot(t_AB3, Y_ABM3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_ABM3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('ABM3', 'FontSize', 17);
%legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])

% BDF3 solution
figure('Name', 'EX 7 - BDF3', 'NumberTitle', 'off');
plot(t_AB3, Y_BDF3(:, 1), '-.', 'LineWidth', 1.5); hold on; grid on;
plot(t_AB3, Y_BDF3(:, 2), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); ylabel('$\mbox{\boldmath $x$}$ [-]', 'FontSize', 18);
title('BDF3', 'FontSize', 17);
legend('$x$', '$y$', 'FontSize', 14, 'Location', 'best');
ylim([-5 5])

% Real system eigenvalues
M = @(t) [-5/2*(1 + 8*sin(t)) 0; 1 1];
lambda = zeros(2, length(t_AB3));
for i = 1:length(t_AB3)
    lambda(:, i) = eig(M(t_AB3(i)))*0.1;
end
figure('Name', 'EX 7 - Real system eigenvalues', 'NumberTitle', 'off');
plot(t_AB3, lambda(2, :), 'LineWidth', 1.5); grid on; hold on;
plot(t_AB3, lambda(1, :), 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 18); ylabel('$h\lambda_i$ [-]', 'FontSize', 18);
title('$h\lambda_i$ of the linearized system', 'FontSize', 17);
legend('$\lambda_x$', '$\lambda_y$', 'FontSize', 14, 'Location','best');

% Stability/Instability domains
figure('Name', 'EX 7 - Stability region', 'NumberTitle', 'off');
re = real(R_AB3);
im = imag(R_AB3);
reL = [re; re(end:-1:1)];
imL = [im; -im(end:-1:1)];
plot(reL, imL, 'LineWIdth', 1.5); hold on; grid on;
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
axis equal; ax = gca; addAxes(ax);
legend({'AB3 Stability margin', 'AM3 Stability Margin',...
    'ABM3 Stability region', 'BDF3 Instability region'}, 'FontSize', 12, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); ylabel('$Im\{h\lambda\}$', 'FontSize', 18);


%% Functions
function [N, H] = getNH(h, tlim)
% getNH - Function to retrieve true N and h values needed for ode operators
% when input value h is not an integer.
%
% PROTOTYPE
%   [N, H] = getNH(h, tlim)
%
% INPUT:
%   h           double [1x1]   step size                                [-]
%   tlim        double [1x2]   time vector limits                       [s]
%
% OUTPUT:
%   N           double [1x1]   modified number of points                [-]
%   H           double [1x1]   modified step size                       [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
%
% -------------------------------------------------------------------------

tVec = tlim(1):h:tlim(2);
N = length(tVec) - 1;
H = tlim(2) - tVec(end);

end


function value = nth_output(N, fcn, varargin)
% nth_output - Useful function to retrieve the wanted output.
%
% PROTOTYPE
%   [value] = nth_output(N, fcn, varargin)
%
% INPUT:
%   N           double       [1x1]   index of wanted output             [-]
%   fcn         fcn hanlde   [1x1]   index of wanted output             [-]
%   varargin    cell         [1xn]   input needed by fcn                [-]
%
% OUTPUT:
%   value       -            [-]     N-th output of function fcn        [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
%
% -------------------------------------------------------------------------
[value{1:N}] = fcn(varargin{:});
value = value{N};
end



function [t, sol, it, fcnE, cV] = secantMethod(f, a, b, toll)
% secantMethod - Function to retrieve the solution of finding zero
% problem of function f performing a secant method algorithm.
%
% PROTOTYPE
%   [t, sol, it, fcnE, cV] = secantMethod(f, a, b, toll)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   a        double      [1x1]   left end of the zero searching range   [-]
%   b        double      [1x1]   right end of the zero searching range  [-]        
%   toll     double      [1x1]   tollerance                             [-]
%
% OUTPUT:
%   t        double      [1x1]   computational time needed              [s]
%   sol      double      [1x1]   solution                               [-]
%   it       double      [1x1]   number of iterations                   [-]
%   fcnE     double      [1x1]   function evaluations                   [-]
%   cV       double      [1xn]   vector of guessed solutions            [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
%
% -------------------------------------------------------------------------

t0  = tic;

err = toll + 1;
it  = 0;

cV = nan(1, 1);

fa = f(a);
fb = f(b);

while err > toll
    c = b - (fb * (b - a))/(fb - fa);
    fc = f(c);
    err = abs(fc);
    
    a = b;
    b = c;
    fa = fb;
    fb = fc;
    
    it = it + 1;
    cV(it) = c;
end

fcnE = it + 2;

sol = cV(it);

t = toc(t0);

end

function [t, sol, it, fcnE, cV] = regulaFalsiMethod(f, a, b, toll)
% regulaFalsiMethod - Function to retrieve the solution of finding zero
% problem of function f performing a regula falsi method algorithm.
%
% PROTOTYPE
%   [t, sol, it, fcnE, cV] = regulaFalsiMethod(f, a, b, toll)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   a        double      [1x1]   left end of the zero searching range   [-]
%   b        double      [1x1]   right end of the zero searching range  [-]        
%   toll     double      [1x1]   tollerance                             [-]
%
% OUTPUT:
%   t        double      [1x1]   computational time needed              [s]
%   sol      double      [1x1]   solution                               [-]
%   it       double      [1x1]   number of iterations                   [-]
%   fcnE     double      [1x1]   function evaluations                   [-]
%   cV       double      [1xn]   vector of guessed solutions            [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
%
% -------------------------------------------------------------------------

t0 = tic;

err = toll + 1;
it   = 0;

cV = nan(1, 1);

fa = f(a);
fb = f(b);

flag = -1;
if fa > 0
    flag = 1;
end

while err > toll
    c = b - (fb * (b - a))/(fb - fa);
    
    fc = f(c);
    err = abs(fc);

    if fc*flag > 0
        a = c;
        fa = fc;
    else
        b = c;
        fb = fc;
    end
    
    it = it + 1;
    cV(it) = c;
    
end

fcnE = it + 2;

sol = cV(it);

t = toc(t0);

end

function [t, sol, it, fcnE, cV] = newt2Da(f, x, invJ, toll)
% newt2Da - Function to compute the zero of the multidimensional
% function described by f applying Newton's method with analytical inverse
% Jacobian matrix described by invJ.
%
% PROTOTYPE
%   [timeComp, sol, it, fcnE, cV] = newt2Da(f, invJ, x, toll)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   x        double      [nx1]   starting guess                         [-]
%   invJ     fcn handle  [1x1]   analytic inverse Jacobian              [-]
%   toll     double      [1x1]   tollerance                             [-]
%
% OUTPUT:
%   t        double      [1x1]   computational time needed              [s]
%   sol      double      [nx1]   solution                               [-]
%   it       double      [1x1]   number of iterations                   [-]
%   fcnE     double      [1x1]   function evaluations                   [-]
%   cV       double      [nxm]   vector of guessed solutions            [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
%
% -------------------------------------------------------------------------

t0 = tic;
err = toll + 1;
it = 0;
cV = nan(size(x));
fx = f(x);

while any(err > toll)
    x = x - invJ(x) * fx;
    fx = f(x);
    err = abs(fx);
    
    it = it + 1;
    cV(:, it) = x;
    
end

fcnE = it + 1;

sol = x;

t = toc(t0);

end


function [t, sol, it, fcnE, cV] = newt2Df(f, x, type, toll)
% newt2Df - Function to compute the zero of the multidimensional
% function described by f applying Newton's method and finite differences
% approximation for the computation of the Jacobian matrix
%
% PROTOTYPE
%   [timeComp, sol, it, fcnE, cV] = newt2Df(f, x, type, toll)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   x        double      [nx1]   starting guess                         [-]
%   type     string      [1x1]   type of finite differences approx.     [-]
%   toll     double      [1x1]   tollerance                             [-]
%
% OUTPUT:
%   t        double      [1x1]   computational time needed              [s]
%   sol      double      [nx1]   solution                               [-]
%   it       double      [1x1]   number of iterations                   [-]
%   fcnE     double      [1x1]   function evaluations                   [-]
%   cV       double      [nxm]   vector of guessed solutions            [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
%
% -------------------------------------------------------------------------
t0 = tic;
err = toll + 1;
it = 0;
cV = nan(size(x));

fx = f(x);

if strcmp(type, 'forward')
    check = true;
else 
    check = false;
end


while any(err > 1e-8)

    delta = sqrt(eps);
    
    if abs(x(1)) > 1
        delta = delta*abs(x(1));
    end
    
    if check
        J(:, 1) = ((f([x(1)+delta, x(2)]) - fx)/delta);
    else
        J(:, 1) = ((f([x(1)+delta, x(2)]) - f([x(1)-delta, x(2)]))/(2*delta));
    end

    delta = sqrt(eps);

    if abs(x(2)) > 1
        delta = delta*abs(x(2));
    end
    
    if check
        J(:, 2) = ((f([x(1), x(2)+delta]) - fx)/delta);
    else
        J(:, 2) = ((f([x(1), x(2)+delta]) - f([x(1), x(2)-delta]))/(2*delta));
    end

    x = x - J\fx;
    fx = f(x);
    err = abs(fx);
    it = it + 1;
    cV(:, it) = x;
    
end
if strcmp(type, 'forward')
    fcnE = 3*it + 1;
else
    fcnE = 5*it + 1;
end

sol = x;

t = toc(t0);

end

function [t, Y, ct, fcnE] = RK1(f, tlim, h, x0)
% RK1 - Function to solve the ODE problem described by f with first order
% Runge-Kutta approximation RK1.
%
% PROTOTYPE
%   [t, Y, ct, fcnE] = RK1(f, tlim, h, x0)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   tlim     double      [1x2]   time vector limits                     [s]
%   h        double      [1x1]   step size                              [-]
%   x0       double      [nx1]   starting value(s)                      [-]
%
% OUTPUT:
%   t        double      [1x1]   time vector                            [s]
%   Y        double      [mxn]   solution                               [-]
%   ct       double      [1x1]   computational time needed              [s]
%   fcnE     double      [1x1]   function evaluations                   [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
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
% RK1 - Function to solve the ODE problem described by f with second order
% Runge-Kutta approximation RK2.
%
% PROTOTYPE
%   [t, Y, ct, fcnE] = RK2(f, tlim, h, x0)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   tlim     double      [1x2]   time vector limits                     [s]
%   h        double      [1x1]   step size                              [-]
%   x0       double      [nx1]   starting value(s)                      [-]
%
% OUTPUT:
%   t        double      [1x1]   time vector                            [s]
%   Y        double      [mxn]   solution                               [-]
%   ct       double      [1x1]   computational time needed              [s]
%   fcnE     double      [1x1]   function evaluations                   [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
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
% RK1 - Function to solve the ODE problem described by f with fourth order
% Runge-Kutta approximation RK4.
%
% PROTOTYPE
%   [t, Y, ct, fcnE] = RK4(f, tlim, h, x0)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   tlim     double      [1x2]   time vector limits                     [s]
%   h        double      [1x1]   step size                              [-]
%   x0       double      [nx1]   starting value(s)                      [-]
%
% OUTPUT:
%   t        double      [1x1]   time vector                            [s]
%   Y        double      [mxn]   solution                               [-]
%   ct       double      [1x1]   computational time needed              [s]
%   fcnE     double      [1x1]   function evaluations                   [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
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

function F = FRK2(h, alpha)
% FRK2 - Function to retrieve the linear operator of the second order
% Runge-Kutta approximation RK2.
%
% PROTOTYPE
%   F = FRK2(h, alpha)
%
% INPUT:
%   h        double     [1x1]   step size                               [-]
%   alpha    double     [1x1]   angle of eigenvalue in the hlambda axis [-]
%
% OUTPUT:
%   F        double     [2x2]   linear operator                         [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
%
% -------------------------------------------------------------------------

A = [0 1; -1 2*cos(alpha)];
F = eye(2) + h*A + 0.5*h^2*A^2;
end

function F = FRK4(h, alpha)
% FRK2 - Function to retrieve the linear operator of the second order
% Runge-Kutta approximation RK4.
%
% PROTOTYPE
%   F = FRK2(h, alpha)
%
% INPUT:
%   h        double     [1x1]   step size                               [-]
%   alpha    double     [1x1]   angle of eigenvalue in the hlambda axis [-]
%
% OUTPUT:
%   F        double     [2x2]   linear operator                         [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
%
% -------------------------------------------------------------------------

A = [0 1; -1 2*cos(alpha)];
F = eye(2) + h*A + 0.5*h^2*A^2 + 1/6*h^3*A^3 + 1/24*h^4*A^4;

end

function [t, Y, ct] = AB3(f, tlim, h, x0)
% AB3 - Function to solve the ODE problem described by f with third order
% Adams Bashfort approximation.
%
% PROTOTYPE
%   [t, Y, ct] = AB3(f, tlim, h, x0)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   tlim     double      [1x2]   time vector limits                     [s]
%   h        double      [1x1]   step size                              [-]
%   x0       double      [nx1]   starting value(s)                      [-]
%
% OUTPUT:
%   t        double      [1x1]   time vector                            [s]
%   Y        double      [mxn]   solution                               [-]
%   ct       double      [1x1]   computational time needed              [s]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
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
% AM3 - Function to solve the ODE problem described by f with third order
% Adams Moulton approximation.
%
% PROTOTYPE
%   [t, Y, ct] = AM3(f, tlim, h, x0)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   tlim     double      [1x2]   time vector limits                     [s]
%   h        double      [1x1]   step size                              [-]
%   x0       double      [nx1]   starting value(s)                      [-]
%
% OUTPUT:
%   t        double      [1x1]   time vector                            [s]
%   Y        double      [mxn]   solution                               [-]
%   ct       double      [1x1]   computational time needed              [s]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
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
                               -(120*h*(Y(2) - 1))/((5*h*Y(1) - 5*h + 12)*(25*h + 200*h*sin(t) + 24)), 12/(5*h*Y(1) - 5*h + 12)];

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
% AM3 - Function to solve the ODE problem described by f with third order
% Adams Bashfort Moulton approximation.
%
% PROTOTYPE
%   [t, Y, ct] = ABM3(f, tlim, h, x0)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   tlim     double      [1x2]   time vector limits                     [s]
%   h        double      [1x1]   step size                              [-]
%   x0       double      [nx1]   starting value(s)                      [-]
%
% OUTPUT:
%   t        double      [1x1]   time vector                            [s]
%   Y        double      [mxn]   solution                               [-]
%   ct       double      [1x1]   computational time needed              [s]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
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
% BDF3 - Function to solve the ODE problem described by f with third order
% backward difference approximation.
%
% PROTOTYPE
%   [t, Y, ct] = BDF3(f, tlim, h, x0)
%
% INPUT:
%   f        fcn handle  [1x1]   function to evaluate                   [-]
%   tlim     double      [1x2]   time vector limits                     [s]
%   h        double      [1x1]   step size                              [-]
%   x0       double      [nx1]   starting value(s)                      [-]
%
% OUTPUT:
%   t        double      [1x1]   time vector                            [s]
%   Y        double      [mxn]   solution                               [-]
%   ct       double      [1x1]   computational time needed              [s]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
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

% Analytical inverse Jacobian
invJ = @(t, Y) [11/(15*h + 120*h*sin(t) + 11), 0; ...
                -(66*h*(Y(2) - 1))/((6*h*Y(1) - 6*h + 11)*(15*h + 120*h*sin(t) + 11)), 11/(6*h*Y(1) - 6*h + 11)];

% Function to be zeroed
fun = @(t, Y, Y1, Y2, Y3) Y - (6/11*h*[-5/2*(1+8*sin(t))*Y(1);(1-Y(1))*Y(2)+Y(1)] + ...
    18/11*Y1 - 9/11*Y2 + 2/11*Y3);

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


function addAxes(ax)
% addAxes - Useful function to plot the X and Y axis with scalatures.
%
% PROTOTYPE
%   [N, H] = getNH(ax)
%
% INPUT:
%   ax          axes [1x1]     handle to current axis                   [-]
%
% OUTPUT: [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
%
% -------------------------------------------------------------------------

line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
l = 1/50 * abs(ax.YLim(2) - ax.YLim(1));

for i = 1:1:length(ax.XTick)
    line([ax.XTick(i) ax.XTick(i)], [0 l], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
end

l = 1/50 * abs(ax.XLim(2) - ax.XLim(1));

for i = 1:1:length(ax.YTick)
    line([0 l], [ax.YTick(i) ax.YTick(i)], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
end

end



function save_fig(fig,name)
% save_fig - Function to save figure to pdf with correct sizing.
%
% PROTOTYPE
%   [t, Y, ct] = BDF3(f, tlim, h, x0)
%
% INPUT:
%   fig      figure      [1x1]   figure to save                         [-]
%   name     double      [1x2]   str                                    [-]
%
% CONTRIBUTOR:
%   Cucchi Lorenzo              10650070
%
% VERSIONS
%   2023-10-30: Release
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