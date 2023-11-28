%% Ex 1
clearvars; close all; clc

% Function to evaluate
func = @(x) [x(2)^2 - x(1) - 2; -x(1)^2 + x(2) + 10];
% Inverse Jacobian of the function
invJ = @(x) (1/(4*x(1)*x(2)-1))*[1, -2*x(2); 2*x(1), -1];
% Tolerance
toll = 1e-8;  

% Finding the first zero z1        
x1 = [1 -4]';
% Analytical method, netwon finite forward, newton finite centered
[sol1a, n1a] = newtAn(func, x1, invJ, toll);
[sol1f, n1f] = newtFF(func, x1, toll);
[sol1c, n1c] = newtFC(func, x1, toll);

% Finding the second zero z2
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
% save_fig(fig,'ex1_1');

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
% save_fig(fig,'ex1_2');


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
% save_fig(fig,'ex1_3');
% system('cd ../Report && pdflatex main.tex > NUL');


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
%   t        - Double [1x1] representing the computational time needed in seconds.
%   sol      - Double [nx1] representing the solution.
%   it       - Double [1x1] indicating the number of iterations.
%   feval     - Double [1x1] representing the number of function evaluations.
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













