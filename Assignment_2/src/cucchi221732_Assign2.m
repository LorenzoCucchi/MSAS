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
clearvars; close all; clc

% Nozzle geometry
geom.L = 0.5;
geom.A = 1;

% Inner layer
thermal.k1 = 5;
geom.l1 = 4e-3;

% Conductive layer
thermal.k2 = 1500;
geom.l2 = 15e-3;
thermal.c2 = 850;
thermal.rho2 = 1880;

% Interface
thermal.k3 = 5;
geom.l3 = 1e-3;

% Insulator
thermal.k4 = 0.07;
geom.l4 = 10e-3;
thermal.c4 = 2100;
thermal.rho4 = 480;

% Outer coating
thermal.k5 = 170;
geom.l5 = 2e-3;

t_v = 0:1e-1:60;
RHS.Ti = @(t) 20+980*t.*(t<=1)+980*(t>1);
RHS.To = 20;
thermal.Tm = 1000;
thermal.To = 20;

thermal.R1 = geom.l1/(geom.A*thermal.k1);
thermal.R2 = geom.l2/(geom.A*thermal.k2);
thermal.R3 = geom.l3/(geom.A*thermal.k3);
thermal.R4 = geom.l4/(geom.A*thermal.k4);
thermal.R5 = geom.l5/(geom.A*thermal.k5);
thermal.C2 = thermal.rho2*geom.l2*geom.A*thermal.c2;
thermal.C4 = thermal.rho4*geom.l4*geom.A*thermal.c4;


ode45_opts = odeset('RelTol',1e-12,'AbsTol',1e-12);

mode.SN.IC = RHS.To*[1 1]';
mode.MN.IC = RHS.To*[1 1 1 1]';

[A,B] = StateSpace(thermal);
mode.SN.eigA = eig(A);
[~,mode.SN.sol] = ode45(@(t,X) A*X+B*[RHS.Ti(t) RHS.To]',t_v,mode.SN.IC,ode45_opts);
mode.SN.sol = Temperatures(mode.SN.sol,thermal,RHS,t_v);

mdl = 'cucchi221732_Assign2sim';

simInp = Simulink.SimulationInput(mdl);
simInp = simInp.setVariable('geom',geom,'Workspace',mdl);
simInp = simInp.setVariable('thermal', thermal,'Workspace',mdl);
out = sim(simInp);


% Plots casual
 fig = figure();
 hold on
 grid minor
 axis padded
 p_nodes = [0 geom.l1/2 (geom.l2/2+geom.l1) (geom.l1+geom.l2)...
     (geom.l1+geom.l2+geom.l4/2) (geom.l1+geom.l2+geom.l4+geom.l5/2)...
     (geom.l1+geom.l2+geom.l4+geom.l5)];
 pp1 = plot(p_nodes,mode.SN.sol(1,:),'ks-','DisplayName','t $=0s$','MarkerSize',9);
 pp2 = plot(p_nodes,mode.SN.sol(6,:),'kd-','DisplayName','t $=0.5s$','MarkerSize',9);
 pp3 = plot(p_nodes,mode.SN.sol(101,:),'k^-','DisplayName','t $=10s$','MarkerSize',9);
 pp4 = plot(p_nodes,mode.SN.sol(201,:),'kv-','DisplayName','t $=20s$','MarkerSize',9);
 pp5 = plot(p_nodes,mode.SN.sol(301,:),'kp-','DisplayName','t $=30s$','MarkerSize',9);
 pp6 = plot(p_nodes,mode.SN.sol(end,:),'k*-','DisplayName','t $=60s$','MarkerSize',9);
 area([0 geom.l1],[1100 1100],'FaceColor','r','FaceAlpha',0.15,'EdgeColor','none');      
 text(0,1050,'Inner lining','Color','r','FontSize',17)
 area([geom.l1 geom.l1+geom.l2],[1100 1100],'FaceColor','b','FaceAlpha',0.15,'EdgeColor','none');      
 text(geom.l2,1050,'Conductor','Color','b','FontSize',17)
 area([geom.l1+geom.l2,  geom.l1+geom.l2+geom.l4],[1100 1100],'FaceColor','g','FaceAlpha',0.15,'EdgeColor','none');      
 text(geom.l1+geom.l2,1050,'Insulator','Color','g','FontSize',17)
 area([geom.l1+geom.l2+geom.l4,  geom.l1+geom.l2+geom.l4+geom.l5],[1100 1100],'FaceColor','m','FaceAlpha',0.15,'EdgeColor','none');      
 text(geom.l1+geom.l2+geom.l4,1050,'Outer coating','Color','m','FontSize',17)
 xlabel('Node')
 ylabel('T [$^\circ$C]')
 legend(pp1.DisplayName,pp2.DisplayName,pp3.DisplayName,pp4.DisplayName,pp5.DisplayName,pp6.DisplayName,'Location','east')
 xticks(p_nodes)
 xticklabels({'i','1','2','3','4','5','o'})
    
 % Plots acasual
 fig = figure();
 hold on
 grid minor
 axis padded
 p_nodes = [0 geom.l1/2 (geom.l2/2+geom.l1) (geom.l1+geom.l2)...
     (geom.l1+geom.l2+geom.l4/2) (geom.l1+geom.l2+geom.l4+geom.l5/2)...
     (geom.l1+geom.l2+geom.l4+geom.l5)];
 pp1 = plot(p_nodes,out.singleSim.signals.values(1,  :),'ks-','DisplayName','t $=0s$','MarkerSize',9);
 pp2 = plot(p_nodes,out.singleSim.signals.values(6, :),'kd-','DisplayName','t $=0.5s$','MarkerSize',9);
 pp3 = plot(p_nodes,out.singleSim.signals.values(101,:),'k^-','DisplayName','t $=10s$','MarkerSize',9);
 pp4 = plot(p_nodes,out.singleSim.signals.values(201,:),'kv-','DisplayName','t $=20s$','MarkerSize',9);
 pp5 = plot(p_nodes,out.singleSim.signals.values(301,:),'kp-','DisplayName','t $=30s$','MarkerSize',9);
 pp6 = plot(p_nodes,out.singleSim.signals.values(end,:),'k*-','DisplayName','t $=60s$','MarkerSize',9);
 area([0 geom.l1],[1100 1100],'FaceColor','r','FaceAlpha',0.15,'EdgeColor','none');      
 text(1.25,1050,'Inner lining','Color','r','FontSize',17)
 area([geom.l1 geom.l1+geom.l2],[1100 1100],'FaceColor','b','FaceAlpha',0.15,'EdgeColor','none');      
 text(2.80,1050,'Conductor','Color','b','FontSize',17)
 area([geom.l1+geom.l2,  geom.l1+geom.l2+geom.l4],[1100 1100],'FaceColor','g','FaceAlpha',0.15,'EdgeColor','none');      
 text(4.40,1050,'Insulator','Color','g','FontSize',17)
 area([geom.l1+geom.l2+geom.l4,  geom.l1+geom.l2+geom.l4+geom.l5],[1100 1100],'FaceColor','m','FaceAlpha',0.15,'EdgeColor','none');      
 text(5.65,1050,'Outer coating','Color','m','FontSize',17)
 xlabel('Node')
 ylabel('T [$^\circ$C]')
 legend(pp1.DisplayName,pp2.DisplayName,pp3.DisplayName,pp4.DisplayName,pp5.DisplayName,pp6.DisplayName,'Location','east')
 xticks(p_nodes)
 xticklabels({'i','1','2','3','4','5','o'})
    
 fig = figure();
 hold on
 grid minor
 axis padded
 p_nodes_m = [0 geom.l1/2 (geom.l2/3+geom.l1) (geom.l2*2/3+geom.l1) (geom.l1+geom.l2)...
     (geom.l1+geom.l2+geom.l4/3) (geom.l1+geom.l2+geom.l4*2/3) (geom.l1+geom.l2+geom.l4+geom.l5/2)...
     (geom.l1+geom.l2+geom.l4+geom.l5)];
 pp1 = plot(p_nodes_m,out.multiSim.signals.values(1, :),'ks-','DisplayName','t $=0s$','MarkerSize',9);
 pp2 = plot(p_nodes_m,out.multiSim.signals.values(6, :),'kd-','DisplayName','t $=0.5s$','MarkerSize',9);
 pp3 = plot(p_nodes_m,out.multiSim.signals.values(101,:),'k^-','DisplayName','t $=10s$','MarkerSize',9);
 pp4 = plot(p_nodes_m,out.multiSim.signals.values(201,:),'kv-','DisplayName','t $=20s$','MarkerSize',9);
 pp5 = plot(p_nodes_m,out.multiSim.signals.values(301,:),'kp-','DisplayName','t $=30s$','MarkerSize',9);
 pp6 = plot(p_nodes_m,out.multiSim.signals.values(end,:),'k*-','DisplayName','t $=60s$','MarkerSize',9);
 area([0 geom.l1],[1100 1100],'FaceColor','r','FaceAlpha',0.15,'EdgeColor','none');      
 text(1.25,1050,'Inner lining','Color','r','FontSize',17)
 area([geom.l1 geom.l1+geom.l2],[1100 1100],'FaceColor','b','FaceAlpha',0.15,'EdgeColor','none');      
 text(2.80,1050,'Conductor','Color','b','FontSize',17)
 area([geom.l1+geom.l2,  geom.l1+geom.l2+geom.l4],[1100 1100],'FaceColor','g','FaceAlpha',0.15,'EdgeColor','none');      
 text(4.40,1050,'Insulator','Color','g','FontSize',17)
 area([geom.l1+geom.l2+geom.l4,  geom.l1+geom.l2+geom.l4+geom.l5],[1100 1100],'FaceColor','m','FaceAlpha',0.15,'EdgeColor','none');      
 text(5.65,1050,'Outer coating','Color','m','FontSize',17)
 xlabel('Node')
 ylabel('T [$^\circ$C]')
 legend(pp1.DisplayName,pp2.DisplayName,pp3.DisplayName,pp4.DisplayName,pp5.DisplayName,pp6.DisplayName,'Location','east')
 xticks(p_nodes_m)
 xticklabels({'i','1','2','3','4','5','6','7','o'})


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
















