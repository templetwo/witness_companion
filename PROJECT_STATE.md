Codex, gpt-6-astra, Jetson seat, 2026-10-12T11:09:09.146747-04:00

# Temple controller: initial capability survey recorded

Temple owns the Pi controller and camera. Vendor controller, phone interface,
button launcher and discovery remain persistently masked; an actual reboot
already verified Temple recovery without vendor fallback. The 4 GB Pi runs the
hardware stack; the Jetson remains the planning host.

See CAPABILITIES.md for the installed vendor baseline, measured/observed results,
and remaining limits. Operator confirmed all eight travel directions, both turns,
small movements/returns of PWM 3–6 and the gripper, plus buzzer and light output.
**All movement was slight.** This is not full-range or useful-speed calibration.
Sonar transaction handling was fixed; camera/battery telemetry and stationary
frozen-driver recovery were tested. Display text, microphone/speaker, IMU and
other optional peripherals remain unverified or unavailable.

Installed vendor data already supplies wheel layout, servo roles, IK geometry,
poses and software offsets. The Pi offsets differ from the downloaded reference.
Next work should adapt that baseline into bounded Temple poses/IK and measure
loaded response, not rediscover nominal geometry through trial motion.

Final release: /opt/temple-witness/releases/20261012-capabilities-04.
52 deployed file hashes verified.145 Jetson tests pass; Pi 144 pass + one skip.
No runtime commissioning override remains. Persistent config records observed
wheel signs but wheel ceiling 0, joints{}, outputs[], sonarfalse. No local arming
signals enabled. Final authenticated status disarmed/inactive; serial released
when disconnected, Temple services active/enabled, vendor units masked/inactive,
battery 7672mV, Pi throttled 0x0, fresh camera JPEG verified.

Detailed evidence: receipts/capability-live-summary.json,
receipts/vendor-mapping-facts.json, receipts/capability-deployed-manifest.json.
No OpenClaw changes, live MCP registration, Stack commit or git push in this task.
Previously authorized push remains blocked by missing GitHub authentication.
