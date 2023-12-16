% Modeling and Simulation of Aerospace System (2023/2024)
% Assignment # 2 es 2
% Author:          Lorenzo Cucchi
% Person code:     10650070
% Student ID:      221732
%
% Comments: every exercise is divide in two parts, the first to solve the
%           exercise questions and the second to plot and print data.
           

clear all
%% Graphics Setup
% Set parameters for desired style of plots
set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'DefaultAxesColorOrder','factory');
%%
clearvars, close all, clc

out = importSim1('out.csv');

%%

time = out.Time;
step.st = 5;
step.y = out.stepy;
voltage.v = out.signalVoltagev;
voltage.i = out.signalVoltagei;
speedw = out.speedSensorw;
gearBox.Q = out.gearboxheatPortQ_flow;
gearBox.T = out.gearboxheatPortT;
gearBox.w = out.gearboxlossyGearw_a;
heatcap.Q = out.heatCapacitorportQ_flow;
heatcap.T = out.heatCapacitorportT;
error = out.feedback;
emfw = out.emfw;

figure()
plot(time,step.y,'--')
hold on
grid on
plot(time,speedw)
xlim([4.5, 6.5])
xlabel("Time [s]")
ylabel("Angular velocity [rad/s]")
legend('step','w','Location','best')

stepinfo(speedw,time-step.st)

figure()
plot(time,voltage.v,'--')
ylabel("Voltage [V]")
hold on
yyaxis right
plot(time,-voltage.i,'-.')
ylabel("Current [A]")
xlim([4.5, 6.5])
xlabel("Time [s]")
grid on
legend('Tension [v]','Current [A]','Location','best')


figure()
plot(time,step.y,'--')
hold on
grid on
plot(time,gearBox.Q)
xlim([4.5, 6.5])
legend('step','Q','Location','best')
xlabel("Time [s]")
ylabel("Power [W]")
stepinfo(gearBox.Q,time-step.st)

figure()
plot(time,gearBox.T-273.15,'--')
hold on
grid on
plot(time,heatcap.T-273.15,'-.')
xlim([0, 120])
xlabel("Time [s]")
ylabel("Temperature [$^\circ$C]")
legend('lossygear','heat capacitor','Location','best')


