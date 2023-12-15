% Modeling and Simulation of Aerospace System (2023/2024)
% Assignment # 2
% Author:          Lorenzo Cucchi
% Person code:     10650070
% Student ID:      221732
%
% Comments: every exercise is divide in two parts, the first to solve the
%           exercise questions and the second to plot and print data.
           


%% Graphics Setup
% Set parameters for desired style of plots
set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLineLineWidth', 1.5);

%% Ex 1
clear all; close all; clc

% Nozzle geometry
geom.L = 0.5;
geom.A = 1;

% Inner layer
thermal.k1 = 1.5;
geom.l1 = 4e-3;

% Conductive layer
thermal.k2 = 1500;
geom.l2 = 10e-3;
thermal.c2 = 750;
thermal.rho2 = 1830;

% Interface
% thermal.k3 = 0.14;
% geom.l3 = 1e-3;
thermal.R3 = 7e-3;

% Insulator
thermal.k4 = 0.07;
geom.l4 = 8e-3;
thermal.c4 = 2100;
thermal.rho4 = 485;

% Outer coating
thermal.k5 = 160;
geom.l5 = 2e-3;

t_v = 0:1e-1:60;
RHS.Ti = @(t) 20+980*t.*(t<=1)+980*(t>1);
RHS.To = 20;
thermal.Tm = 1000;
thermal.To = 20;

thermal.R1 = geom.l1/(geom.A*thermal.k1);
thermal.R2 = geom.l2/(geom.A*thermal.k2);
% thermal.R3 = geom.l3/(geom.A*thermal.k3);
thermal.R4 = geom.l4/(geom.A*thermal.k4);
thermal.R5 = geom.l5/(geom.A*thermal.k5);
thermal.C2 = thermal.rho2*geom.l2*geom.A*thermal.c2;
thermal.C4 = thermal.rho4*geom.l4*geom.A*thermal.c4;


ode45_opts = odeset('RelTol',1e-12,'AbsTol',1e-12);

casual.IC = RHS.To*[1 1]';

[A,B] = StateSpace(thermal);

% Eigenvalue calculation
eigA = eig(A);
fprintf("The matrix A eigenvalues are: [%0.3f,%0.3f]\n\n",eigA)
[~,casual.sol] = ode45(@(t,X) A*X+B*[RHS.Ti(t) RHS.To]',t_v,casual.IC,ode45_opts);

casual.sol = Temperatures(casual.sol,thermal,RHS,t_v);

% Simscape solution
mdl = 'cucchi221732_Assign2sim';

simInp = Simulink.SimulationInput(mdl);
simInp = simInp.setVariable('geom',geom,'Workspace',mdl);
simInp = simInp.setVariable('thermal', thermal,'Workspace',mdl);
simIn = setModelParameter(simInp,"StopTime","60",'FixedStep','0.1');
% start simulation
out = sim(simIn);


error = (out.singleSim.signals.values-casual.sol)./(casual.sol)*100;
% Plots casual
 fig = figure();
 hold on
 grid minor
 axis padded
 set(0,'DefaultAxesColorOrder',copper(6))
 p_nodes = [0 geom.l1/2 (geom.l2/2+geom.l1) (geom.l1+geom.l2)...
     (geom.l1+geom.l2+geom.l4/2) (geom.l1+geom.l2+geom.l4+geom.l5/2)...
     (geom.l1+geom.l2+geom.l4+geom.l5)];
 pp1 = plot(p_nodes,casual.sol(1,:),'s-','DisplayName','t $=0s$','MarkerSize',9);
 pp2 = plot(p_nodes,casual.sol(6,:),'d-','DisplayName','t $=0.5s$','MarkerSize',9);
 pp3 = plot(p_nodes,casual.sol(101,:),'o-','DisplayName','t $=10s$','MarkerSize',9);
 pp4 = plot(p_nodes,casual.sol(201,:),'^-','DisplayName','t $=20s$','MarkerSize',9);
 pp5 = plot(p_nodes,casual.sol(301,:),'v-','DisplayName','t $=30s$','MarkerSize',9);
 pp6 = plot(p_nodes,casual.sol(end,:),'>-','DisplayName','t $=60s$','MarkerSize',9);   
 xline(0,'--')
 xline(geom.l1,'--')    
 xline([geom.l1+geom.l2],'--')
 xline([geom.l1+geom.l2+geom.l4],'--')
 xline([geom.l1+geom.l2+geom.l4+geom.l5],'--')
 xlabel('Node')
 ylabel('T [$^\circ$C]')
 legend(pp1.DisplayName,pp2.DisplayName,pp3.DisplayName,pp4.DisplayName,pp5.DisplayName,pp6.DisplayName,'Location','east')
 xticks(p_nodes)
 xticklabels({'i','1','2','3','4','5','o'})
 save_fig(fig,'ex1-1')
 % Plots acasual
 fig = figure();
 hold on
 grid minor
 axis padded
 set(0,'DefaultAxesColorOrder',copper(6))
 p_nodes = [0 geom.l1/2 (geom.l2/2+geom.l1) (geom.l1+geom.l2)...
     (geom.l1+geom.l2+geom.l4/2) (geom.l1+geom.l2+geom.l4+geom.l5/2)...
     (geom.l1+geom.l2+geom.l4+geom.l5)];
 pp1 = plot(p_nodes,out.singleSim.signals.values(1,  :),'s-','DisplayName','t $=0s$','MarkerSize',9);
 pp2 = plot(p_nodes,out.singleSim.signals.values(6, :),'d-','DisplayName','t $=0.5s$','MarkerSize',9);
 pp3 = plot(p_nodes,out.singleSim.signals.values(101,:),'o-','DisplayName','t $=10s$','MarkerSize',9);
 pp4 = plot(p_nodes,out.singleSim.signals.values(201,:),'^-','DisplayName','t $=20s$','MarkerSize',9);
 pp5 = plot(p_nodes,out.singleSim.signals.values(301,:),'v-','DisplayName','t $=30s$','MarkerSize',9);
 pp6 = plot(p_nodes,out.singleSim.signals.values(end,:),'>-','DisplayName','t $=60s$','MarkerSize',9);
 xline(0,'--')
 xline(geom.l1,'--')    
 xline([geom.l1+geom.l2],'--')
 xline([geom.l1+geom.l2+geom.l4],'--')
 xline([geom.l1+geom.l2+geom.l4+geom.l5],'--')
 xlabel('Node')
 ylabel('T [$^\circ$C]')
 legend(pp1.DisplayName,pp2.DisplayName,pp3.DisplayName,pp4.DisplayName,pp5.DisplayName,pp6.DisplayName,'Location','east')
 xticks(p_nodes)
 xticklabels({'i','1','2','3','4','5','o'})
 save_fig(fig,'ex1-2')

 fig = figure();
 hold on
 grid minor
 axis padded
 set(0,'DefaultAxesColorOrder',copper(6))
 p_nodes_m = [0 geom.l1/2 (geom.l2/3+geom.l1) (geom.l2*2/3+geom.l1) (geom.l1+geom.l2)...
     (geom.l1+geom.l2+geom.l4/3) (geom.l1+geom.l2+geom.l4*2/3) (geom.l1+geom.l2+geom.l4+geom.l5/2)...
     (geom.l1+geom.l2+geom.l4+geom.l5)];
 pp1 = plot(p_nodes_m,out.multiSim.signals.values(1, :),'s-','DisplayName','t $=0s$','MarkerSize',9);
 pp2 = plot(p_nodes_m,out.multiSim.signals.values(6, :),'d-','DisplayName','t $=0.5s$','MarkerSize',9);
 pp3 = plot(p_nodes_m,out.multiSim.signals.values(101,:),'o-','DisplayName','t $=10s$','MarkerSize',9);
 pp4 = plot(p_nodes_m,out.multiSim.signals.values(201,:),'^-','DisplayName','t $=20s$','MarkerSize',9);
 pp5 = plot(p_nodes_m,out.multiSim.signals.values(301,:),'v-','DisplayName','t $=30s$','MarkerSize',9);
 pp6 = plot(p_nodes_m,out.multiSim.signals.values(end,:),'>-','DisplayName','t $=60s$','MarkerSize',9);
 xline(0,'--')
 xline(geom.l1,'--')    
 xline([geom.l1+geom.l2],'--')
 xline([geom.l1+geom.l2+geom.l4],'--')
 xline([geom.l1+geom.l2+geom.l4+geom.l5],'--')
 xlabel('Node')
 ylabel('T [$^\circ$C]')
 legend(pp1.DisplayName,pp2.DisplayName,pp3.DisplayName,pp4.DisplayName,pp5.DisplayName,pp6.DisplayName,'Location','east')
 xticks(p_nodes_m)
 xticklabels({'i','1','2','3','4','5','6','7','o'})
 save_fig(fig,'ex1-3')
 
 fig = figure()
 hold on
 grid minor
 axis padded
 color = copper(9);
 pp1 = plot(out.tout,out.multiSim.signals.values(:,1),'Color',color(9,:),'DisplayName','Node i','MarkerSize',9);
 pp2 = plot(out.tout,out.multiSim.signals.values(:,2),'Color',color(8,:),'DisplayName','Node 1','MarkerSize',9);
 pp3 = plot(out.tout,out.multiSim.signals.values(:,3),'Color',color(7,:),'DisplayName','Node 2','MarkerSize',9);
 pp4 = plot(out.tout,out.multiSim.signals.values(:,4),'Color',color(6,:),'DisplayName','Node 3','MarkerSize',9);
 pp5 = plot(out.tout,out.multiSim.signals.values(:,5),'Color',color(5,:),'DisplayName','Node 4','MarkerSize',9);
 pp6 = plot(out.tout,out.multiSim.signals.values(:,6),'Color',color(4,:),'DisplayName','Node 5','MarkerSize',9);
 pp7 = plot(out.tout,out.multiSim.signals.values(:,7),'Color',color(3,:),'DisplayName','Node 6','MarkerSize',9);
 pp8 = plot(out.tout,out.multiSim.signals.values(:,8),'Color',color(2,:),'DisplayName','Node 7','MarkerSize',9);
 pp9 = plot(out.tout,out.multiSim.signals.values(:,9),'Color',color(1,:),'DisplayName','Node o','MarkerSize',9);
 xlabel('Time [s]')
 ylabel('T [$^\circ$C]')
 legend(pp1.DisplayName,pp2.DisplayName,pp3.DisplayName,pp4.DisplayName,...
     pp5.DisplayName,pp6.DisplayName,pp7.DisplayName,pp8.DisplayName,pp9.DisplayName,'Location','east')
 save_fig(fig,'ex1-4')

 fig = figure()
 hold on
 grid minor
 axis padded
 color = copper(7);
 pp1 = plot(out.tout,error(:,2),'Color',color(6,:),'DisplayName','Node 1','MarkerSize',9);
 pp2 = plot(out.tout,error(:,3),'Color',color(5,:),'DisplayName','Node 2','MarkerSize',9);
 pp3 = plot(out.tout,error(:,4),'Color',color(4,:),'DisplayName','Node 3','MarkerSize',9);
 pp4 = plot(out.tout,error(:,5),'Color',color(3,:),'DisplayName','Node 4','MarkerSize',9);
 pp5 = plot(out.tout,error(:,6),'Color',color(2,:),'DisplayName','Node 5','MarkerSize',9);
 xlabel('Time [s]')
 ylabel('Difference [\%]')
 legend(pp1.DisplayName,pp2.DisplayName,pp3.DisplayName,pp4.DisplayName,...
     pp5.DisplayName,'Location','east')
save_fig(fig,'ex1-5')

function [A,B] = StateSpace(thermal)

A1n = -(thermal.R1+thermal.R2+thermal.R3+thermal.R4/2);
A1d = (thermal.C2*(thermal.R1+thermal.R2/2)*(thermal.R2/2+thermal.R3+thermal.R4/2));
A2n = 1;
A2d = (thermal.C2*(thermal.R2/2+thermal.R3+thermal.R4/2));
A3n = 1;
A3d = (thermal.C4*(thermal.R2/2+thermal.R3+thermal.R4/2));
A4n = -(thermal.R2/2+thermal.R3+thermal.R4+thermal.R5);
A4d = (thermal.C4*(thermal.R2/2+thermal.R3+thermal.R4/2)*(thermal.R4/2+thermal.R5));

B1 = 1/(thermal.C2*(thermal.R1+thermal.R2/2)); 
B4 = 1/(thermal.C4*(thermal.R4/2+thermal.R5));


% State-space matrices
A = [A1n/A1d, A2n/A2d; A3n/A3d, A4n/A4d];
B = [B1, 0; 0, B4];

end

function T = Temperatures(T,thermal,RHS,t_vect)

 
 Ti = RHS.Ti(t_vect)';
 To = RHS.To; 

 T2 = T(:,1);    
 T4 = T(:,2);

 Q1 = (Ti-T2)/(thermal.R1+thermal.R2/2);             
 T1 = Ti-thermal.R1/2*Q1;

 Q2 = (T2-T4)/(thermal.R2/2+thermal.R3+thermal.R4/2);        
 T3 = T2-(thermal.R2/2+thermal.R3/2)*Q2;

 Q3 = (T4-To)/(thermal.R4/2+thermal.R5);             
 T5 = To+thermal.R5/2*Q3;

 T(:,1) = Ti;    
 T(:,2) = T1;    
 T(:,3) = T2;    
 T(:,4) = T3;
 T(:,5) = T4;    
 T(:,6) = T5;    
 T(:,7) = To;

end













