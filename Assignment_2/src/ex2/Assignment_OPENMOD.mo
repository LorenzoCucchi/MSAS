model Assignment
  Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage annotation(
    Placement(transformation(origin = {98, 24}, extent = {{10, -10}, {-10, 10}})));
  Modelica.Electrical.Analog.Basic.Ground ground annotation(
    Placement(transformation(origin = {-138, 2}, extent = {{208, -8}, {220, 4}})));
  Modelica.Electrical.Analog.Basic.Resistor resistor(R = 0.1) annotation(
    Placement(transformation(origin = {114, 6}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
  Modelica.Electrical.Analog.Basic.Inductor inductor(L = 0.01) annotation(
    Placement(transformation(origin = {114, -18}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
  Modelica.Electrical.Analog.Basic.RotationalEMF emf(k = 0.3) annotation(
    Placement(transformation(origin = {114, -46}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Mechanics.Rotational.Components.Inertia inertia(J = 0.001) annotation(
    Placement(transformation(origin = {-138, 2}, extent = {{270, -58}, {290, -38}})));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(transformation(origin = {-138, 2}, extent = {{246, -80}, {258, -68}})));
  Modelica.Mechanics.Rotational.Sensors.SpeedSensor speedSensor annotation(
    Placement(transformation(origin = {88, -98}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Mechanics.Rotational.Components.Gearbox gearbox(useSupport = false, ratio = 2, lossTable = [0, 0.99, 0.99, 0, 0; 50, 0.98, 0.98, 0.5, 0.5; 100, 0.97, 0.97, 1, 1; 210, 0.96, 0.96, 1.5, 1.5], useHeatPort = true) annotation(
    Placement(transformation(origin = {-138, 2}, extent = {{306, -58}, {326, -38}})));
  Modelica.Mechanics.Rotational.Components.Inertia Propeller(J = 1.09, phi(start = 0), w(start = 0), a(start = 0)) annotation(
    Placement(transformation(origin = {216, -46}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Mechanics.Rotational.Sources.QuadraticSpeedDependentTorque quadraticSpeedDependentTorque(tau_nominal = -100, TorqueDirection = false, w_nominal = 210) annotation(
    Placement(transformation(origin = {256, -46}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Blocks.Sources.Step step(height = 210, offset = 0, startTime = 5) annotation(
    Placement(transformation(origin = {-138, 2}, extent = {{60, -80}, {80, -60}})));
  Modelica.Blocks.Continuous.PI PI(k = 0.14, T = 0.1, initType = Modelica.Blocks.Types.Init.InitialState, x_start = 0) annotation(
    Placement(transformation(origin = {-2, -68}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Math.Feedback feedback annotation(
    Placement(transformation(origin = {-34, -68}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor thermalConductor(G = 100) annotation(
    Placement(transformation(origin = {168, -70}, extent = {{-8, -8}, {8, 8}}, rotation = 270)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C = 3000) annotation(
    Placement(transformation(origin = {194, -84}, extent = {{-4, -4}, {4, 4}}, rotation = 270)));
  Modelica.Blocks.Nonlinear.Limiter limiter(uMax = 200, uMin = 0, strict = false) annotation(
    Placement(transformation(origin = {-138, 2}, extent = {{160, -80}, {180, -60}})));
  Modelica.Blocks.Sources.Constant const3(k = -4000) annotation(
    Placement(transformation(origin = {-102, -150}, extent = {{-10, -10}, {10, 10}}, rotation = -0)));
  Modelica.Thermal.HeatTransfer.Celsius.TemperatureSensor temperatureSensor annotation(
    Placement(transformation(origin = {196, -156}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Logical.Switch switch1 annotation(
    Placement(transformation(origin = {220, -286}, extent = {{-10, 10}, {10, -10}}, rotation = 180)));
  Modelica.Thermal.FluidHeatFlow.Components.OpenTank openTank2(ATank = 0.01, T0(displayUnit = "K") = 283.15, T0fixed = true, hTank = 0.8, level(fixed = true, start = 1e-12), medium = Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(), pAmbient = 100000, useHeatPort = true) annotation(
    Placement(transformation(origin = {192, -342}, extent = {{-152, 4}, {-172, 24}}, rotation = -0)));
  Modelica.Blocks.Sources.Constant const1(k = 2.0*0.008/300) annotation(
    Placement(transformation(origin = {286, -268}, extent = {{10, -10}, {-10, 10}})));
  Modelica.Blocks.Sources.Constant const2(k = 0) annotation(
    Placement(transformation(origin = {286, -304}, extent = {{10, -10}, {-10, 10}})));
  Modelica.Blocks.Logical.Hysteresis hysteresisT1(pre_y_start = true, uHigh = 273.15 + 10, uLow = 273.15 + 5) annotation(
    Placement(transformation(origin = {-196, -252}, extent = {{172, 16}, {152, 36}}, rotation = -0)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow annotation(
    Placement(transformation(origin = {28, -174}, extent = {{10, -10}, {-10, 10}}, rotation = -180)));
  Modelica.Blocks.Logical.Hysteresis hysteresisT(pre_y_start = false, uHigh = 54, uLow = 46) annotation(
    Placement(transformation(origin = {132, -212}, extent = {{88, 46}, {108, 66}})));
  Modelica.Blocks.Sources.Constant const(k = 300) annotation(
    Placement(transformation(origin = {210, -192}, extent = {{10, -10}, {-10, 10}})));
  Modelica.Thermal.FluidHeatFlow.Components.Pipe pipe(T0(displayUnit = "K") = 283.15, T0fixed = true, dpLaminar = 0, dpNominal = 0, frictionLoss = 0, h_g = 0, m = 1000*0.4*0.02^2*Modelica.Constants.pi, medium = Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(), useHeatPort = true) annotation(
    Placement(transformation(origin = {168, -246}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Thermal.HeatTransfer.Components.Convection convection annotation(
    Placement(transformation(origin = {168, -192}, extent = {{-10, -10}, {10, 10}}, rotation = 270)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow1 annotation(
    Placement(transformation(origin = {22, -292}, extent = {{8, -8}, {-8, 8}}, rotation = -180)));
  Modelica.Thermal.FluidHeatFlow.Sources.VolumeFlow volumeFlow(T0(displayUnit = "K") = 283.15, T0fixed = true, V_flow(start = 0), constantVolumeFlow = 0, medium = Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(), useVolumeFlowInput = true) annotation(
    Placement(transformation(origin = {124, -246}, extent = {{-10, 10}, {10, -10}})));
  Modelica.Blocks.Logical.Switch switch2 annotation(
    Placement(transformation(origin = {-34, -174}, extent = {{10, 10}, {-10, -10}}, rotation = -180)));
  Modelica.Blocks.Sources.Constant const4(k = 0) annotation(
    Placement(transformation(origin = {-100, -198}, extent = {{-10, -10}, {10, 10}}, rotation = -0)));
  Modelica.Thermal.FluidHeatFlow.Components.OpenTank openTank(ATank = 0.01, T0(displayUnit = "K") = 283.15, T0fixed = true, hTank = 0.8, level(fixed = true, start = 0.8), medium = Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(), pAmbient = 100000, useHeatPort = true) annotation(
    Placement(transformation(origin = {-166, -234}, extent = {{210, 4}, {190, 24}}, rotation = -0)));
  Modelica.Blocks.Sources.Constant constant1(k = 0) annotation(
    Placement(transformation(origin = {-94, -316}, extent = {{-10, -10}, {10, 10}}, rotation = -0)));
  Modelica.Blocks.Sources.Constant constant2(k = -4000) annotation(
    Placement(transformation(origin = {-92, -272}, extent = {{-10, -10}, {10, 10}}, rotation = -0)));
  Modelica.Blocks.Logical.Switch switch annotation(
    Placement(transformation(origin = {-32, -292}, extent = {{10, 10}, {-10, -10}}, rotation = -180)));
  Modelica.Blocks.Logical.Hysteresis hysteresis(pre_y_start = true, uHigh = 273.15 + 8, uLow = 273.15 + 6) annotation(
    Placement(transformation(origin = {-194, -360}, extent = {{172, 16}, {152, 36}}, rotation = -0)));
equation
  connect(ground.p, signalVoltage.n) annotation(
    Line(points = {{76, 6}, {76, 24}, {88, 24}}, color = {0, 0, 255}));
  connect(signalVoltage.p, resistor.n) annotation(
    Line(points = {{108, 24}, {110, 24}, {110, 16}, {114, 16}}, color = {0, 0, 255}));
  connect(inductor.n, resistor.p) annotation(
    Line(points = {{114, -8}, {114, -4}}, color = {0, 0, 255}));
  connect(inductor.p, emf.p) annotation(
    Line(points = {{114, -28}, {114, -36}}, color = {0, 0, 255}));
  connect(emf.flange, inertia.flange_a) annotation(
    Line(points = {{124, -46}, {132, -46}}));
  connect(ground1.p, emf.n) annotation(
    Line(points = {{114, -66}, {114, -56}}, color = {0, 0, 255}));
  connect(inertia.flange_b, gearbox.flange_a) annotation(
    Line(points = {{152, -46}, {168, -46}}));
  connect(gearbox.flange_b, Propeller.flange_a) annotation(
    Line(points = {{188, -46}, {206, -46}}));
  connect(quadraticSpeedDependentTorque.flange, Propeller.flange_b) annotation(
    Line(points = {{246, -46}, {226, -46}}));
  connect(speedSensor.flange, Propeller.flange_b) annotation(
    Line(points = {{98, -98}, {232, -98}, {232, -46}, {226, -46}}));
  connect(feedback.y, PI.u) annotation(
    Line(points = {{-25, -68}, {-14, -68}}, color = {0, 0, 127}));
  connect(step.y, feedback.u1) annotation(
    Line(points = {{-57, -68}, {-42, -68}}, color = {0, 0, 127}));
  connect(feedback.u2, speedSensor.w) annotation(
    Line(points = {{-34, -76}, {-34, -98}, {77, -98}}, color = {0, 0, 127}));
  connect(thermalConductor.port_b, heatCapacitor.port) annotation(
    Line(points = {{168, -78}, {168, -84}, {190, -84}}, color = {191, 0, 0}));
  connect(thermalConductor.port_a, gearbox.heatPort) annotation(
    Line(points = {{168, -62}, {168, -56}}, color = {191, 0, 0}));
  connect(PI.y, limiter.u) annotation(
    Line(points = {{9, -68}, {20, -68}}, color = {0, 0, 127}));
  connect(limiter.y, signalVoltage.v) annotation(
    Line(points = {{43, -68}, {66, -68}, {66, 42}, {98, 42}, {98, 36}}, color = {0, 0, 127}));
  connect(prescribedHeatFlow.port, openTank.heatPort) annotation(
    Line(points = {{38, -174}, {89, -174}, {89, -230}, {44, -230}}, color = {191, 0, 0}));
  connect(const.y, convection.Gc) annotation(
    Line(points = {{199, -192}, {178, -192}}, color = {0, 0, 127}));
  connect(switch1.y, volumeFlow.volumeFlow) annotation(
    Line(points = {{209, -286}, {124, -286}, {124, -256}}, color = {0, 0, 127}));
  connect(prescribedHeatFlow.Q_flow, switch2.y) annotation(
    Line(points = {{18, -174}, {-23, -174}}, color = {0, 0, 127}));
  connect(pipe.flowPort_a, openTank2.flowPort) annotation(
    Line(points = {{178, -246}, {178, -245}, {194, -245}, {194, -352}, {30, -352}, {30, -338}}, color = {255, 0, 0}));
  connect(hysteresisT1.y, switch2.u2) annotation(
    Line(points = {{-45, -226}, {-74, -226}, {-74, -174}, {-46, -174}}, color = {255, 0, 255}));
  connect(convection.fluid, pipe.heatPort) annotation(
    Line(points = {{168, -202}, {168, -236}}, color = {191, 0, 0}));
  connect(const4.y, switch2.u3) annotation(
    Line(points = {{-89, -198}, {-80.5, -198}, {-80.5, -182}, {-46, -182}}, color = {0, 0, 127}));
  connect(const3.y, switch2.u1) annotation(
    Line(points = {{-91, -150}, {-77.5, -150}, {-77.5, -166}, {-46, -166}}, color = {0, 0, 127}));
  connect(prescribedHeatFlow1.port, openTank2.heatPort) annotation(
    Line(points = {{30, -292}, {90, -292}, {90, -338.5}, {40, -338.5}, {40, -338}}, color = {191, 0, 0}));
  connect(const1.y, switch1.u1) annotation(
    Line(points = {{275, -268}, {238, -268}, {238, -278}, {232, -278}}, color = {0, 0, 127}));
  connect(openTank.flowPort, volumeFlow.flowPort_a) annotation(
    Line(points = {{34, -230}, {34.5, -230}, {34.5, -246}, {114, -246}}, color = {255, 0, 0}));
  connect(hysteresisT.u, temperatureSensor.T) annotation(
    Line(points = {{218, -156}, {206, -156}}, color = {0, 0, 127}));
  connect(hysteresisT1.u, openTank.TTank) annotation(
    Line(points = {{-22, -226}, {23, -226}}, color = {0, 0, 127}));
  connect(hysteresisT.y, switch1.u2) annotation(
    Line(points = {{241, -156}, {254, -156}, {254, -286}, {232, -286}}, color = {255, 0, 255}));
  connect(const2.y, switch1.u3) annotation(
    Line(points = {{275, -304}, {240, -304}, {240, -294}, {232, -294}}, color = {0, 0, 127}));
  connect(volumeFlow.flowPort_b, pipe.flowPort_b) annotation(
    Line(points = {{134, -246}, {158, -246}}, color = {255, 0, 0}));
  connect(convection.solid, thermalConductor.port_b) annotation(
    Line(points = {{168, -182}, {168, -78}}, color = {191, 0, 0}));
  connect(temperatureSensor.port, thermalConductor.port_b) annotation(
    Line(points = {{186, -156}, {168, -156}, {168, -78}}, color = {191, 0, 0}));
  connect(constant1.y, switch.u3) annotation(
    Line(points = {{-83, -316}, {-58, -316}, {-58, -300}, {-44, -300}}, color = {0, 0, 127}));
  connect(constant2.y, switch.u1) annotation(
    Line(points = {{-81, -272}, {-48.5, -272}, {-48.5, -284}, {-44, -284}}, color = {0, 0, 127}));
  connect(hysteresis.y, switch.u2) annotation(
    Line(points = {{-43, -334}, {-66, -334}, {-66, -292}, {-44, -292}}, color = {255, 0, 255}));
  connect(switch.y, prescribedHeatFlow1.Q_flow) annotation(
    Line(points = {{-21, -292}, {14, -292}}, color = {0, 0, 127}));
  connect(openTank2.TTank, hysteresis.u) annotation(
    Line(points = {{19, -334}, {-20, -334}}, color = {0, 0, 127}));
  annotation(
    Icon(coordinateSystem(preserveAspectRatio = false, extent = {{-120, -360}, {480, 100}})),
    Diagram(coordinateSystem(preserveAspectRatio = false, extent = {{-120, -360}, {480, 100}})),
    uses(Modelica(version = "4.0.0")));
end Assignment;
