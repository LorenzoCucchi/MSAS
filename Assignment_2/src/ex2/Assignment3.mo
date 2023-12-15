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
  Modelica.Thermal.FluidHeatFlow.Components.Pipe pipe1(medium = Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(), m = 1000*0.4*0.02^2*Modelica.Constants.pi, T0(displayUnit = "K") = 280.65, T0fixed = true, V_flow(start = 0), dpLaminar = 0, dpNominal = 0, frictionLoss = 0, useHeatPort = true, h_g = 0) annotation(
    Placement(transformation(origin = {302, -222}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Thermal.HeatTransfer.Components.Convection convection1 annotation(
    Placement(transformation(origin = {302, -172}, extent = {{-10, -10}, {10, 10}}, rotation = 270)));
  Modelica.Blocks.Sources.Constant const5(k = 300) annotation(
    Placement(transformation(origin = {340, -172}, extent = {{10, -10}, {-10, 10}})));
  Modelica.Thermal.FluidHeatFlow.Sources.VolumeFlow volumeFlow1(medium = Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(), T0(displayUnit = "K") = 280.65, T0fixed = true, V_flow(start = 0), useVolumeFlowInput = true, constantVolumeFlow = 0) annotation(
    Placement(transformation(origin = {202, -222}, extent = {{-10, 10}, {10, -10}})));
  Modelica.Thermal.HeatTransfer.Celsius.TemperatureSensor temperatureSensor1 annotation(
    Placement(transformation(origin = {328, -144}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Logical.Hysteresis hysteresisT2(uLow = 42, uHigh = 57, pre_y_start = false) annotation(
    Placement(transformation(origin = {-562, -48}, extent = {{920, -106}, {940, -86}})));
  Modelica.Blocks.Logical.Switch switch3 annotation(
    Placement(transformation(origin = {394, -252}, extent = {{-10, 10}, {10, -10}}, rotation = 180)));
  Modelica.Blocks.Sources.Constant const6(k = 2.5*0.008/300) annotation(
    Placement(transformation(origin = {450, -232}, extent = {{10, -10}, {-10, 10}})));
  Modelica.Blocks.Sources.Constant const7(k = -4500) annotation(
    Placement(transformation(origin = {-102, -180}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Logical.Hysteresis hysteresisT3(pre_y_start = true, uHigh = 273.15 + 9, uLow = 273.15 + 6) annotation(
    Placement(transformation(origin = {-124, -218}, extent = {{172, 16}, {152, 36}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow2 annotation(
    Placement(transformation(origin = {100, -140}, extent = {{10, -10}, {-10, 10}}, rotation = -180)));
  Modelica.Blocks.Logical.Switch switch4 annotation(
    Placement(transformation(origin = {38, -140}, extent = {{10, 10}, {-10, -10}}, rotation = -180)));
  Modelica.Blocks.Sources.Constant const8(k = 0) annotation(
    Placement(transformation(origin = {-102, -236}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.FluidHeatFlow.Components.OpenTank openTank1(ATank = 0.01, T0(displayUnit = "K") = 280.65, T0fixed = true, hTank = 0.8, level(fixed = true, start = 0.8), medium = Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(), pAmbient = 100000, useHeatPort = true) annotation(
    Placement(transformation(origin = {-94, -200}, extent = {{210, 4}, {190, 24}})));
  Modelica.Thermal.FluidHeatFlow.Components.OpenTank openTank3(ATank = 0.01, T0(displayUnit = "K") = 280.65, T0fixed = true, hTank = 0.8, level(fixed = true, start = 0.00000000000001), medium = Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(), pAmbient = 100000, useHeatPort = true) annotation(
    Placement(transformation(origin = {260, -306}, extent = {{-152, 4}, {-172, 24}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow3 annotation(
    Placement(transformation(origin = {90, -256}, extent = {{8, -8}, {-8, 8}}, rotation = -180)));
  Modelica.Blocks.Logical.Switch switch5 annotation(
    Placement(transformation(origin = {36, -256}, extent = {{10, 10}, {-10, -10}}, rotation = -180)));
  Modelica.Blocks.Logical.Hysteresis hysteresis1(pre_y_start = false, uHigh = 273.15 + 9, uLow = 273.15 + 6) annotation(
    Placement(transformation(origin = {-128, -324}, extent = {{172, 16}, {152, 36}})));
  Modelica.Blocks.Math.Product product1 annotation(
    Placement(transformation(origin = {278, -258}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Blocks.Math.Feedback feedback1 annotation(
    Placement(transformation(origin = {-562, -48}, extent = {{742, -266}, {762, -246}})));
  Modelica.Blocks.Logical.Hysteresis hysteresis2(pre_y_start = true, uHigh = 0.7999999, uLow = -0.7999999) annotation(
    Placement(transformation(origin = {382, -330}, extent = {{-172, 16}, {-152, 36}})));
  Modelica.Blocks.Math.BooleanToReal booleanToReal(realTrue = 1, realFalse = -1) annotation(
    Placement(transformation(origin = {-562, -48}, extent = {{820, -266}, {840, -246}})));
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
  connect(convection1.fluid, pipe1.heatPort) annotation(
    Line(points = {{302, -182}, {302, -212}}, color = {191, 0, 0}));
  connect(const5.y, convection1.Gc) annotation(
    Line(points = {{329, -172}, {312, -172}}, color = {0, 0, 127}));
  connect(hysteresisT2.u, temperatureSensor1.T) annotation(
    Line(points = {{356, -144}, {338, -144}}, color = {0, 0, 127}));
  connect(hysteresisT2.y, switch3.u2) annotation(
    Line(points = {{379, -144}, {474, -144}, {474, -252}, {406, -252}}, color = {255, 0, 255}));
  connect(const6.y, switch3.u1) annotation(
    Line(points = {{439, -232}, {414, -232}, {414, -244}, {406, -244}}, color = {0, 0, 127}));
  connect(volumeFlow1.flowPort_b, pipe1.flowPort_b) annotation(
    Line(points = {{212, -222}, {292, -222}}, color = {255, 0, 0}));
  connect(prescribedHeatFlow2.port, openTank1.heatPort) annotation(
    Line(points = {{110, -140}, {161, -140}, {161, -196}, {116, -196}}, color = {191, 0, 0}));
  connect(prescribedHeatFlow2.Q_flow, switch4.y) annotation(
    Line(points = {{90, -140}, {49, -140}}, color = {0, 0, 127}));
  connect(hysteresisT3.y, switch4.u2) annotation(
    Line(points = {{27, -192}, {-2, -192}, {-2, -140}, {26, -140}}, color = {255, 0, 255}));
  connect(const8.y, switch4.u3) annotation(
    Line(points = {{-91, -236}, {6, -236}, {6, -148}, {26, -148}}, color = {0, 0, 127}));
  connect(const7.y, switch4.u1) annotation(
    Line(points = {{-91, -180}, {-50, -180}, {-50, -132}, {26, -132}}, color = {0, 0, 127}));
  connect(openTank1.TTank, hysteresisT3.u) annotation(
    Line(points = {{95, -192}, {50, -192}}, color = {0, 0, 127}));
  connect(prescribedHeatFlow3.port, openTank3.heatPort) annotation(
    Line(points = {{98, -256}, {162, -256}, {162, -302.5}, {108, -302.5}, {108, -302}}, color = {191, 0, 0}));
  connect(hysteresis1.y, switch5.u2) annotation(
    Line(points = {{23, -298}, {2, -298}, {2, -256}, {24, -256}}, color = {255, 0, 255}));
  connect(switch5.y, prescribedHeatFlow3.Q_flow) annotation(
    Line(points = {{47, -256}, {82, -256}}, color = {0, 0, 127}));
  connect(openTank3.TTank, hysteresis1.u) annotation(
    Line(points = {{87, -298}, {46, -298}}, color = {0, 0, 127}));
  connect(openTank1.flowPort, volumeFlow1.flowPort_a) annotation(
    Line(points = {{106, -196}, {106, -222}, {192, -222}}, color = {255, 0, 0}));
  connect(switch5.u3, const8.y) annotation(
    Line(points = {{24, -264}, {24, -268}, {-74, -268}, {-74, -236}, {-91, -236}}, color = {0, 0, 127}));
  connect(switch5.u1, switch4.u1) annotation(
    Line(points = {{24, -248}, {-50, -248}, {-50, -132}, {26, -132}}, color = {0, 0, 127}));
  connect(pipe1.flowPort_a, openTank3.flowPort) annotation(
    Line(points = {{312, -222}, {354, -222}, {354, -338}, {98, -338}, {98, -302}}, color = {255, 0, 0}));
  connect(switch3.y, product1.u2) annotation(
    Line(points = {{383, -252}, {290, -252}}, color = {0, 0, 127}));
  connect(volumeFlow1.volumeFlow, product1.y) annotation(
    Line(points = {{202, -232}, {202, -258}, {267, -258}}, color = {0, 0, 127}));
  connect(openTank1.level, feedback1.u1) annotation(
    Line(points = {{95, -186}, {90, -186}, {90, -234}, {172, -234}, {172, -304}, {182, -304}}, color = {0, 0, 127}));
  connect(openTank3.level, feedback1.u2) annotation(
    Line(points = {{87, -292}, {82, -292}, {82, -324}, {190, -324}, {190, -312}}, color = {0, 0, 127}));
  connect(feedback1.y, hysteresis2.u) annotation(
    Line(points = {{199, -304}, {208, -304}}, color = {0, 0, 127}));
  connect(switch3.u3, const8.y) annotation(
    Line(points = {{406, -260}, {416, -260}, {416, -354}, {-74, -354}, {-74, -236}, {-91, -236}}, color = {0, 0, 127}));
  connect(hysteresis2.y, booleanToReal.u) annotation(
    Line(points = {{231, -304}, {256, -304}}, color = {255, 0, 255}));
  connect(product1.u1, booleanToReal.y) annotation(
    Line(points = {{290, -264}, {298, -264}, {298, -304}, {279, -304}}, color = {0, 0, 127}));
  connect(convection1.solid, thermalConductor.port_b) annotation(
    Line(points = {{302, -162}, {302, -118}, {168, -118}, {168, -78}}, color = {191, 0, 0}));
  connect(temperatureSensor1.port, thermalConductor.port_b) annotation(
    Line(points = {{318, -144}, {302.5, -144}, {302.5, -138}, {303, -138}, {303, -140}, {302, -140}, {302, -117}, {168, -117}, {168, -78}}, color = {191, 0, 0}));
  annotation(
    Icon(coordinateSystem(preserveAspectRatio = false, extent = {{-120, -360}, {480, 100}})),
    Diagram(coordinateSystem(preserveAspectRatio = false, extent = {{-120, -360}, {480, 100}})),
    uses(Modelica(version = "4.0.0")));
end Assignment;
