within ;
model Assignment2
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C=3000, T(
      displayUnit="degC",
      start=358.15,
      fixed=true))
    annotation (Placement(transformation(extent={{22,78},{42,98}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow fixedHeatFlow(
    Q_flow=2414.5,
    T_ref=358.15,
    alpha=0)
    annotation (Placement(transformation(extent={{-78,44},{-54,68}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor thermalConductor(G=100)
             annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={-6,56})));
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
        origin={32,-22})));
  Modelica.Thermal.HeatTransfer.Components.Convection convection
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=270,
        origin={32,28})));
  Modelica.Blocks.Sources.Constant const(k=300) annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=0,
        origin={66,28})));
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
        origin={-68,-22})));
  Modelica.Thermal.HeatTransfer.Celsius.TemperatureSensor
                                         temperatureSensor annotation (
      Placement(transformation(
        origin={56,56},
        extent={{-10,-10},{10,10}},
        rotation=0)));
  Modelica.Blocks.Logical.Hysteresis hysteresisT(
    uLow=42,
    uHigh=59,
    pre_y_start=false)
    annotation (Placement(transformation(extent={{88,46},{108,66}})));
  Modelica.Blocks.Logical.Switch switch1 annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=180,
        origin={124,-52})));
  Modelica.Blocks.Sources.Constant const1(k=0.75*0.008/300) annotation (
      Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=0,
        origin={170,-30})));
  Modelica.Blocks.Sources.Constant const7(k=-2800)   annotation (
    Placement(transformation(origin={-340,68},      extent = {{-10, -10}, {10, 10}}, rotation = -0)));
  Modelica.Blocks.Logical.Hysteresis hysteresisT3(
    pre_y_start=true,
    uHigh=273.15 + 9,
    uLow=273.15 + 6)                                                                                          annotation (
    Placement(transformation(origin={-394,-18},     extent = {{172, 16}, {152, 36}}, rotation = -0)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow2
                                                                              annotation (
    Placement(transformation(origin={-170,60},    extent = {{10, -10}, {-10, 10}}, rotation = -180)));
  Modelica.Blocks.Logical.Switch switch4 annotation (
    Placement(transformation(origin={-232,60},     extent = {{10, 10}, {-10, -10}}, rotation = -180)));
  Modelica.Blocks.Sources.Constant const8(k=0)   annotation (
    Placement(transformation(origin={-338,-38},     extent = {{-10, -10}, {10, 10}}, rotation = -0)));
  Modelica.Thermal.FluidHeatFlow.Components.OpenTank openTank1(
    ATank=0.01,
    T0(displayUnit="K") = 280.65,
    T0fixed=true,
    hTank=0.8,
    level(fixed=true, start=0.8),
    medium=Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(),
    pAmbient=100000,
    useHeatPort=true)                                                                                                                                                                                                         annotation (
    Placement(transformation(origin={-364,0},       extent = {{210, 4}, {190, 24}}, rotation = -0)));
  Modelica.Thermal.FluidHeatFlow.Components.OpenTank openTank3(
    ATank=0.01,
    T0(displayUnit="K") = 280.65,
    T0fixed=true,
    hTank=0.8,
    level(fixed=true, start=0.000000001),
    medium=Modelica.Thermal.FluidHeatFlow.Media.Water_10degC(),
    pAmbient=100000,
    useHeatPort=true)                                                                                                                                                                                                         annotation (
    Placement(transformation(origin={-10,-106},    extent = {{-152, 4}, {-172, 24}}, rotation = -0)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow3 annotation (
    Placement(transformation(origin={-172,-56},   extent = {{8, -8}, {-8, 8}}, rotation = -180)));
  Modelica.Blocks.Logical.Switch switch annotation (
    Placement(transformation(origin={-234,-56},    extent = {{10, 10}, {-10, -10}}, rotation = -180)));
  Modelica.Blocks.Logical.Hysteresis hysteresis(
    pre_y_start=false,
    uHigh=273.15 + 9,
    uLow=273.15 + 6)                                                                                       annotation (
    Placement(transformation(origin={-398,-124},    extent = {{172, 16}, {152, 36}}, rotation = -0)));
  Modelica.Blocks.Math.Product product1 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={8,-58})));
  Modelica.Blocks.Math.Feedback feedback
    annotation (Placement(transformation(extent={{-90,-114},{-70,-94}})));
  Modelica.Blocks.Logical.Hysteresis hysteresis1(
    pre_y_start=true,
    uHigh=0.7999999,
    uLow=-0.7999999)                                                                                       annotation (
    Placement(transformation(origin={112,-130},     extent={{-172,16},{-152,36}},    rotation = -0)));
  Modelica.Blocks.Math.BooleanToReal booleanToReal(realTrue=1, realFalse=-1)
    annotation (Placement(transformation(extent={{-12,-114},{8,-94}})));
equation
  connect(heatCapacitor.port,thermalConductor. port_b)
    annotation (Line(points={{32,78},{32,56},{4,56}},  color={191,0,0}));
  connect(fixedHeatFlow.port,thermalConductor. port_a)
    annotation (Line(points={{-54,56},{-16,56}}, color={191,0,0}));
  connect(convection.fluid,pipe. heatPort) annotation (Line(points={{32,18},{32,
          -12}},
        color={191,0,0}));
  connect(const.y,convection. Gc)
    annotation (Line(points={{55,28},{42,28}}, color={0,0,127}));
  connect(convection.solid,thermalConductor. port_b) annotation (Line(points={{32,38},
          {32,56},{4,56}},                         color={191,0,0}));
  connect(temperatureSensor.port,thermalConductor. port_b)
    annotation (Line(points={{46,56},{4,56}},   color={191,0,0}));
  connect(hysteresisT.u,temperatureSensor. T)
    annotation (Line(points={{86,56},{66,56}}, color={0,0,127}));
  connect(hysteresisT.y,switch1. u2) annotation (Line(points={{109,56},{198,56},
          {198,-52},{136,-52}},
                              color={255,0,255}));
  connect(const1.y,switch1. u1) annotation (Line(points={{159,-30},{148,-30},{
          148,-44},{136,-44}},
                           color={0,0,127}));
  connect(volumeFlow.flowPort_b,pipe. flowPort_b) annotation (Line(points={{-58,-22},
          {22,-22}},                                            color={255,0,0}));
  connect(prescribedHeatFlow2.port, openTank1.heatPort) annotation (Line(points=
         {{-160,60},{-109,60},{-109,4},{-154,4}}, color={191,0,0}));
  connect(prescribedHeatFlow2.Q_flow, switch4.y)
    annotation (Line(points={{-180,60},{-221,60}}, color={0,0,127}));
  connect(hysteresisT3.y,switch4. u2) annotation (
    Line(points={{-243,8},{-272,8},{-272,60},{-244,60}},                color = {255, 0, 255}));
  connect(const8.y,switch4. u3) annotation (
    Line(points={{-327,-38},{-270,-38},{-270,52},{-244,52}},                color = {0, 0, 127}));
  connect(const7.y,switch4. u1) annotation (
    Line(points={{-329,68},{-244,68}},                                      color = {0, 0, 127}));
  connect(openTank1.TTank, hysteresisT3.u)
    annotation (Line(points={{-175,8},{-220,8}}, color={0,0,127}));
  connect(prescribedHeatFlow3.port,openTank3. heatPort) annotation (
    Line(points={{-164,-56},{-108,-56},{-108,-102.5},{-162,-102.5},{-162,-102}},    color = {191, 0, 0}));
  connect(hysteresis.y,switch. u2) annotation (
    Line(points={{-247,-98},{-268,-98},{-268,-56},{-246,-56}},          color = {255, 0, 255}));
  connect(switch.y,prescribedHeatFlow3. Q_flow) annotation (
    Line(points={{-223,-56},{-180,-56}},     color = {0, 0, 127}));
  connect(openTank3.TTank,hysteresis. u) annotation (
    Line(points={{-183,-98},{-224,-98}},     color = {0, 0, 127}));
  connect(openTank1.flowPort, volumeFlow.flowPort_a)
    annotation (Line(points={{-164,4},{-164,-22},{-78,-22}}, color={255,0,0}));
  connect(switch.u3, const8.y) annotation (Line(points={{-246,-64},{-304,-64},{
          -304,-38},{-327,-38}},            color={0,0,127}));
  connect(switch.u1, switch4.u1) annotation (Line(points={{-246,-48},{-288,-48},
          {-288,68},{-244,68}}, color={0,0,127}));
  connect(pipe.flowPort_a, openTank3.flowPort) annotation (Line(points={{42,-22},
          {84,-22},{84,-138},{-172,-138},{-172,-102}}, color={255,0,0}));
  connect(switch1.y, product1.u2)
    annotation (Line(points={{113,-52},{20,-52}}, color={0,0,127}));
  connect(volumeFlow.volumeFlow, product1.y)
    annotation (Line(points={{-68,-32},{-68,-58},{-3,-58}}, color={0,0,127}));
  connect(openTank1.level, feedback.u1) annotation (Line(points={{-175,14},{
          -180,14},{-180,-34},{-98,-34},{-98,-104},{-88,-104}}, color={0,0,127}));
  connect(openTank3.level, feedback.u2) annotation (Line(points={{-183,-92},{
          -188,-92},{-188,-124},{-80,-124},{-80,-112}}, color={0,0,127}));
  connect(feedback.y, hysteresis1.u)
    annotation (Line(points={{-71,-104},{-62,-104}}, color={0,0,127}));
  connect(switch1.u3, const8.y) annotation (Line(points={{136,-60},{164,-60},{
          164,-160},{-304,-160},{-304,-38},{-327,-38}}, color={0,0,127}));
  connect(hysteresis1.y, booleanToReal.u)
    annotation (Line(points={{-39,-104},{-14,-104}}, color={255,0,255}));
  connect(product1.u1, booleanToReal.y) annotation (Line(points={{20,-64},{28,
          -64},{28,-104},{9,-104}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-420,-180},{220,
            120}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-420,-180},{
            220,120}})),
    uses(Modelica(version="4.0.0")),
    experiment(
      StopTime=1000,
      __Dymola_NumberOfIntervals=100000,
      __Dymola_Algorithm="Dassl"));
end Assignment2;
