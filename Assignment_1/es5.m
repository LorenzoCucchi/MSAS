%% EX 5
clearvars; close all; clc; warning off;

% Statement of the problem
A = @(alpha) [0 1; -1 2*cos(alpha)];
alpha_v = linspace(0, pi, 1000);
theta_v = [0.4 0.1 0.3 0.7 0.9];

% Linear operator derived function of h alpha and theta
F_BI = @(h, a, th) (eye(2) - (1 - th)*h*A(a) + ...
    ((((1 - th)*h)^2)/2)*A(a)^2)\(eye(2) + th*h*A(a) + (((th*h)^2)/2)*A(a)^2);

h_v = zeros(length(alpha_v),length(theta_v));
R_v = zeros(length(alpha_v),length(theta_v));
re = zeros(length(alpha_v),length(theta_v));
im = zeros(length(alpha_v),length(theta_v));
reL = zeros(length(alpha_v)*2,length(theta_v));
imL = zeros(length(alpha_v)*2,length(theta_v));
alphaL = cell(length(theta_v), 1);
RL = cell(length(theta_v), 1);

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
xLIM = xlim; 
xlim([xLIM(1)-0.5 xLIM(2)+0.5]);
axis equal; 
ax = gca; 
line([0 0], ax.YLim, 'color', [0 0 0 0.5], 'LineWidth', 0.1);
line(ax.XLim, [0 0], 'color', [0 0 0 0.5], 'LineWidth', 0.1);
legend({'$\theta=0.4$','$\theta=0.1$','$\theta=0.3$','$\theta=0.7$','$\theta=0.9$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$Re\{h\lambda\}$', 'FontSize', 18); 
ylabel('$Im\{h\lambda\}$', 'FontSize', 18);
title('Unstable regions of $BI2_{\theta}$', 'FontSize', 17);
% save_fig(fig,'ex5_1');

% h solutions
fig = figure();
hold on; 
grid on;
for i = 1:length(theta_v)
   plot(alphaL{i}, abs(RL{i}),'LineWidth', 1.5);
end
xlim([0 180]);
legend({'$\theta=0.4$','$\theta=0.1$','$\theta=0.3$','$\theta=0.7$','$\theta=0.9$'}, 'FontSize', 14, 'Location', 'best');
xlabel('$\alpha$ [deg]', 'FontSize', 18); 
ylabel('$h$ [-]', 'FontSize', 18);
title('Minimum $h$ for stability', 'FontSize', 17);


