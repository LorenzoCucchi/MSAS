within ;
model Assignment
  Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage annotation (
      Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=0,
        origin={236,20})));
  Modelica.Electrical.Analog.Basic.Ground ground
    annotation (Placement(transformation(extent={{210,-80},{222,-68}})));
  Modelica.Electrical.Analog.Basic.Resistor resistor(R=0.1) annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={252,4})));
  Modelica.Electrical.Analog.Basic.Inductor inductor(i(start=0, fixed=true),
                                                     L=0.01) annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={252,-20})));
  Modelica.Electrical.Analog.Basic.RotationalEMF emf(k=0.3) annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={252,-48})));
  Modelica.Mechanics.Rotational.Components.Inertia inertia(J=0.001,
    phi(start=0, fixed=true),
    w(start=0, fixed=true))
    annotation (Placement(transformation(extent={{270,-58},{290,-38}})));
  Modelica.Mechanics.Rotational.Sensors.SpeedSensor speedSensor annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={226,-100})));
  Modelica.Mechanics.Rotational.Components.Inertia Propeller(
    J=0.0535,
    phi(start=0),
    w(start=0, fixed=true),
    a(start=0))
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={344,-48})));
  Modelica.Mechanics.Rotational.Sources.QuadraticSpeedDependentTorque
    quadraticSpeedDependentTorque(
    tau_nominal=-100,
    TorqueDirection=false,
    w_nominal=210)                                                annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={380,-48})));
  Modelica.Blocks.Sources.Step step(
    height=210,
    offset=0,
    startTime=5)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=0,
        origin={136,-94})));
  Modelica.Blocks.Math.Feedback feedback annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={160,-78})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor thermalConductor(G=100)
             annotation (Placement(transformation(
        extent={{-8,-8},{8,8}},
        rotation=0,
        origin={314,-88})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C=3000, T(start=
          293.15, fixed=true))
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={344,-78})));
  Modelica.Blocks.Nonlinear.Limiter limiter(
    uMax=200,
    uMin=0,
    strict=false)
    annotation (Placement(transformation(extent={{200,32},{216,48}})));
  Modelica.Mechanics.Rotational.Components.LossyGear
                                  lossyGear(
    final ratio=2,
    final lossTable=[0,0.99,0.99,0,0; 50,0.98,0.98,0.5,0.5; 100,0.97,0.97,1,1;
        210,0.96,0.96,1.5,1.5],
    final useSupport=false,
    final useHeatPort=true) annotation (Placement(transformation(extent={{304,-58},
            {324,-38}})));
  Modelica.Blocks.Continuous.Integrator
                               I(
    k=7.0,
    use_reset=false,
    initType=Modelica.Blocks.Types.Init.NoInit)
    "Integral part of PID controller"
    annotation (Placement(transformation(extent={{-8,-8},{8,8}},
        rotation=90,
        origin={160,-26})));
  Modelica.Blocks.Continuous.Derivative
                               D(
    k=0.017,
    T=0.00017,
    x_start=0,
    initType=Modelica.Blocks.Types.Init.NoInit)
                             "Derivative part of PID controller"
    annotation (Placement(transformation(extent={{-8,-8},{8,8}},
        rotation=90,
        origin={186,-26})));
  Modelica.Blocks.Math.Add3
                   Add(k3=-1)
                       annotation (Placement(transformation(extent={{-8,-8},{8,
            8}},
        rotation=90,
        origin={160,26})));
  Modelica.Blocks.Math.Gain
                   P1(k=0.15)
                          "Proportional part of PID controller"
    annotation (Placement(transformation(extent={{-8,-8},{8,8}},
        rotation=90,
        origin={136,-26})));
equation
  connect(ground.p, signalVoltage.n) annotation (
    Line(points={{216,-68},{216,20},{226,20}},      color = {0, 0, 255}));
  connect(signalVoltage.p, resistor.n) annotation (
    Line(points={{246,20},{252,20},{252,14}},                   color = {0, 0, 255}));
  connect(inductor.n, resistor.p) annotation (
    Line(points = {{252, -10}, {252, -6}}, color = {0, 0, 255}));
  connect(inductor.p, emf.p) annotation (
    Line(points = {{252, -30}, {252, -38}}, color = {0, 0, 255}));
  connect(emf.flange, inertia.flange_a) annotation (
    Line(points = {{262, -48}, {270, -48}}, color = {0, 0, 0}));
  connect(quadraticSpeedDependentTorque.flange, Propeller.flange_b) annotation (
    Line(points={{370,-48},{354,-48}},      color = {0, 0, 0}));
  connect(speedSensor.flange, Propeller.flange_b) annotation (
    Line(points={{236,-100},{360,-100},{360,-48},{354,-48}},          color = {0, 0, 0}));
  connect(step.y, feedback.u1) annotation (
    Line(points={{147,-94},{160,-94},{160,-86}},
                                          color = {0, 0, 127}));
  connect(feedback.u2, speedSensor.w) annotation (
    Line(points={{168,-78},{186,-78},{186,-100},{215,-100}},
                                                          color = {0, 0, 127}));
  connect(thermalConductor.port_b, heatCapacitor.port) annotation (
    Line(points={{322,-88},{344,-88}},                  color = {191, 0, 0}));
  connect(limiter.y, signalVoltage.v) annotation (
    Line(points={{216.8,40},{236,40},{236,32}},                              color = {0, 0, 127}));
  connect(inertia.flange_b, lossyGear.flange_a)
    annotation (Line(points={{290,-48},{304,-48}}, color={0,0,0}));
  connect(lossyGear.flange_b, Propeller.flange_a)
    annotation (Line(points={{324,-48},{334,-48}}, color={0,0,0}));
  connect(thermalConductor.port_a, lossyGear.heatPort)
    annotation (Line(points={{306,-88},{300,-88},{300,-58},{304,-58}},
                                                   color={191,0,0}));
  connect(emf.n, ground.p)
    annotation (Line(points={{252,-58},{252,-68},{216,-68}}, color={0,0,255}));
  connect(I.y,Add. u2)
    annotation (Line(points={{160,-17.2},{160,16.4}},
                                              color={0,0,127}));
  connect(D.y, Add.u3) annotation (Line(points={{186,-17.2},{186,16.4},{166.4,
          16.4}}, color={0,0,127}));
  connect(Add.y, limiter.u)
    annotation (Line(points={{160,34.8},{160,40},{198.4,40}},
                                                   color={0,0,127}));
  connect(Add.u1, P1.y) annotation (Line(points={{153.6,16.4},{136,16.4},{136,
          -17.2}}, color={0,0,127}));
  connect(D.u, speedSensor.w) annotation (Line(points={{186,-35.6},{186,-100},{
          215,-100}}, color={0,0,127}));
  connect(P1.u, I.u) annotation (Line(points={{136,-35.6},{136,-54},{160,-54},{
          160,-35.6}}, color={0,0,127}));
  connect(feedback.y, I.u)
    annotation (Line(points={{160,-69},{160,-35.6}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{120,-120},{400,60}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{120,-120},{400,
            60}})),
    uses(Modelica(version="4.0.0")),
    experiment(
      StopTime=120,
      __Dymola_NumberOfIntervals=12000,
      Tolerance=1e-12,
      __Dymola_Algorithm="Esdirk45a"));
end Assignment;
