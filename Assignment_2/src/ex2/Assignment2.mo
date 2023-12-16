within ;
model Assignment2
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C=3000, T(
      displayUnit="degC",
      start=358.15,
      fixed=true))
    annotation (Placement(transformation(extent={{8,58},{28,78}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow fixedHeatFlow(
    Q_flow=2414.5,
    T_ref=358.15,
    alpha=0)
    annotation (Placement(transformation(extent={{-12,-12},{12,12}},
        rotation=0,
        origin={-52,42})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor thermalConductor(G=100)
             annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={-10,42})));
  Modelica.Thermal.FluidHeatFlow.Components.Pipe pipe(
    medium=Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(),
    m=1000*0.4*0.02^2*Modelica.Constants.pi,
    T0(displayUnit="K") = 280.65,
    T0fixed=true,
    V_flow(start=0),
    dpLaminar=0,
    dpNominal=0,
    frictionLoss=0,
    useHeatPort=true,
    h_g=0) annotation (Placement(transformation(extent={{-10,-10},{10,10}},
          rotation=180,
        origin={18,-22})));
  Modelica.Thermal.HeatTransfer.Components.Convection convection
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=270,
        origin={18,8})));
  Modelica.Blocks.Sources.Constant const(k=300) annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=0,
        origin={54,8})));
  Modelica.Thermal.FluidHeatFlow.Sources.VolumeFlow volumeFlow(
    medium=Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(),
    m=1000*0.4*0.02^2*Modelica.Constants.pi,
    T0(displayUnit="K") = 280.65,
    T0fixed=true,
    V_flow(start=0, fixed=true),
    useVolumeFlowInput=true,
    constantVolumeFlow=0) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=0,
        origin={-34,-22})));
  Modelica.Thermal.HeatTransfer.Celsius.TemperatureSensor
                                         temperatureSensor annotation (
      Placement(transformation(
        origin={38,42},
        extent={{-10,-10},{10,10}},
        rotation=0)));
  Modelica.Blocks.Logical.Hysteresis hysteresisT(
    uLow=42,
    uHigh=59,
    pre_y_start=false)
    annotation (Placement(transformation(extent={{62,32},{82,52}})));
  Modelica.Blocks.Logical.Switch switchF annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=180,
        origin={64,-52})));
  Modelica.Blocks.Sources.Constant const1(k=0.75*0.008/300) annotation (
      Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=0,
        origin={102,-44})));
  Modelica.Blocks.Sources.Constant constRef(k=-2800) annotation (Placement(
        transformation(
        origin={-204,60},
        extent={{-10,-10},{10,10}},
        rotation=-0)));
  Modelica.Blocks.Logical.Hysteresis hysteresisT1(
    pre_y_start=true,
    uHigh=273.15 + 9,
    uLow=273.15 + 6)                                                                                          annotation (
    Placement(transformation(origin={-312,-18},     extent = {{172, 16}, {152, 36}}, rotation = -0)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow2
                                                                              annotation (
    Placement(transformation(origin={-108,60},    extent={{10,-10},{-10,10}},      rotation = -180)));
  Modelica.Blocks.Logical.Switch switchT1 annotation (Placement(transformation(
        origin={-150,60},
        extent={{10,10},{-10,-10}},
        rotation=-180)));
  Modelica.Blocks.Sources.Constant constOff(k=0) annotation (Placement(
        transformation(
        origin={-204,-34},
        extent={{-10,-10},{10,10}},
        rotation=-0)));
  Modelica.Thermal.FluidHeatFlow.Components.OpenTank openTank1(
    ATank=0.01,
    T0(displayUnit="K") = 280.65,
    T0fixed=true,
    hTank=0.8,
    level(fixed=true, start=0.8),
    medium=Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(),
    pAmbient=100000,
    useHeatPort=true)                                                                                                                                                                                                         annotation (
    Placement(transformation(origin={-292,0},       extent = {{210, 4}, {190, 24}}, rotation = -0)));
  Modelica.Thermal.FluidHeatFlow.Components.OpenTank openTank2(
    ATank=0.01,
    T0(displayUnit="K") = 280.65,
    T0fixed=true,
    hTank=0.8,
    level(fixed=true, start=0.0001),
    medium=Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(),
    pAmbient=100000,
    useHeatPort=true)                                                                                                                                                                                                         annotation (
    Placement(transformation(origin={60,-106},     extent = {{-152, 4}, {-172, 24}}, rotation = -0)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow3 annotation (
    Placement(transformation(origin={-110,-60},   extent={{8,-8},{-8,8}},      rotation = -180)));
  Modelica.Blocks.Logical.Switch switchT2 annotation (Placement(transformation(
        origin={-150,-60},
        extent={{10,10},{-10,-10}},
        rotation=-180)));
  Modelica.Blocks.Logical.Hysteresis hysteresisT2(
    pre_y_start=false,
    uHigh=273.15 + 9,
    uLow=273.15 + 6) annotation (Placement(transformation(
        origin={-312,-124},
        extent={{172,16},{152,36}},
        rotation=-0)));
  Modelica.Blocks.Math.Product product1 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={8,-58})));
  Modelica.Blocks.Math.Feedback feedback
    annotation (Placement(transformation(extent={{-56,-100},{-36,-80}})));
  Modelica.Blocks.Logical.Hysteresis hysteresisF(
    pre_y_start=true,
    uHigh=0.799,
    uLow=-0.799)                                                                                           annotation (
    Placement(transformation(origin={148,-116},     extent={{-172,16},{-152,36}},    rotation = -0)));
  Modelica.Blocks.Math.BooleanToReal booleanToReal(realTrue=1, realFalse=-1)
    annotation (Placement(transformation(extent={{10,-100},{30,-80}})));
equation
  connect(heatCapacitor.port,thermalConductor. port_b)
    annotation (Line(points={{18,58},{18,42},{0,42}},  color={191,0,0}));
  connect(fixedHeatFlow.port,thermalConductor. port_a)
    annotation (Line(points={{-40,42},{-20,42}}, color={191,0,0}));
  connect(convection.fluid,pipe. heatPort) annotation (Line(points={{18,-2},{18,
          -12}},
        color={191,0,0}));
  connect(const.y,convection. Gc)
    annotation (Line(points={{43,8},{28,8}},   color={0,0,127}));
  connect(convection.solid,thermalConductor. port_b) annotation (Line(points={{18,18},
          {18,42},{0,42}},                         color={191,0,0}));
  connect(temperatureSensor.port,thermalConductor. port_b)
    annotation (Line(points={{28,42},{0,42}},   color={191,0,0}));
  connect(hysteresisT.u,temperatureSensor. T)
    annotation (Line(points={{60,42},{48,42}}, color={0,0,127}));
  connect(hysteresisT.y,switchF. u2) annotation (Line(points={{83,42},{86,42},{
          86,-52},{76,-52}},  color={255,0,255}));
  connect(const1.y,switchF. u1) annotation (Line(points={{91,-44},{76,-44}},
                           color={0,0,127}));
  connect(volumeFlow.flowPort_b,pipe. flowPort_b) annotation (Line(points={{-24,-22},
          {8,-22}},                                             color={255,0,0}));
  connect(prescribedHeatFlow2.port, openTank1.heatPort) annotation (Line(points={{-98,60},
          {-74,60},{-74,4},{-82,4}},              color={191,0,0}));
  connect(prescribedHeatFlow2.Q_flow, switchT1.y)
    annotation (Line(points={{-118,60},{-139,60}}, color={0,0,127}));
  connect(hysteresisT1.y, switchT1.u2) annotation (Line(points={{-161,8},{-174,
          8},{-174,60},{-162,60}}, color={255,0,255}));
  connect(constOff.y, switchT1.u3) annotation (Line(points={{-193,-34},{-168,
          -34},{-168,52},{-162,52}}, color={0,0,127}));
  connect(constRef.y, switchT1.u1) annotation (Line(points={{-193,60},{-180,60},
          {-180,68},{-162,68}}, color={0,0,127}));
  connect(openTank1.TTank,hysteresisT1. u)
    annotation (Line(points={{-103,8},{-138,8}}, color={0,0,127}));
  connect(prescribedHeatFlow3.port,openTank2. heatPort) annotation (
    Line(points={{-102,-60},{-74,-60},{-74,-102},{-92,-102}},                       color = {191, 0, 0}));
  connect(hysteresisT2.y, switchT2.u2) annotation (Line(points={{-161,-98},{
          -174,-98},{-174,-60},{-162,-60}}, color={255,0,255}));
  connect(switchT2.y, prescribedHeatFlow3.Q_flow)
    annotation (Line(points={{-139,-60},{-118,-60}}, color={0,0,127}));
  connect(openTank2.TTank, hysteresisT2.u)
    annotation (Line(points={{-113,-98},{-138,-98}}, color={0,0,127}));
  connect(openTank1.flowPort, volumeFlow.flowPort_a)
    annotation (Line(points={{-92,4},{-92,-22},{-44,-22}},   color={255,0,0}));
  connect(switchT2.u3, constOff.y) annotation (Line(points={{-162,-68},{-184,
          -68},{-184,-34},{-193,-34}}, color={0,0,127}));
  connect(switchT2.u1, switchT1.u1) annotation (Line(points={{-162,-52},{-180,
          -52},{-180,68},{-162,68}}, color={0,0,127}));
  connect(pipe.flowPort_a,openTank2. flowPort) annotation (Line(points={{28,-22},
          {50,-22},{50,-108},{-102,-108},{-102,-102}}, color={255,0,0}));
  connect(switchF.y, product1.u2)
    annotation (Line(points={{53,-52},{20,-52}},  color={0,0,127}));
  connect(volumeFlow.volumeFlow, product1.y)
    annotation (Line(points={{-34,-32},{-34,-58},{-3,-58}}, color={0,0,127}));
  connect(openTank1.level, feedback.u1) annotation (Line(points={{-103,14},{
          -108,14},{-108,-28},{-66,-28},{-66,-90},{-54,-90}},   color={0,0,127}));
  connect(openTank2.level, feedback.u2) annotation (Line(points={{-113,-92},{
          -122,-92},{-122,-112},{-46,-112},{-46,-98}},  color={0,0,127}));
  connect(feedback.y,hysteresisF. u)
    annotation (Line(points={{-37,-90},{-26,-90}},   color={0,0,127}));
  connect(switchF.u3, constOff.y) annotation (Line(points={{76,-60},{86,-60},{
          86,-120},{-184,-120},{-184,-34},{-193,-34}}, color={0,0,127}));
  connect(hysteresisF.y, booleanToReal.u)
    annotation (Line(points={{-3,-90},{8,-90}},      color={255,0,255}));
  connect(product1.u1, booleanToReal.y) annotation (Line(points={{20,-64},{40,
          -64},{40,-90},{31,-90}},  color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-220,-120},{120,
            80}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-220,-120},{
            120,80}})),
    uses(Modelica(version="4.0.0")),
    experiment(
      StopTime=1000,
      __Dymola_NumberOfIntervals=100000,
      __Dymola_Algorithm="Dassl"));
end Assignment2;
