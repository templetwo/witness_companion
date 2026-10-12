Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:06:14 EDT

# Source findings A and B

Read-only reference root: /home/tony/vendor/MasterPi-src/. References below are
to that tree. Source inspection is not proof of the code installed on the Pi.
No vendor imports or source files were included in this repository.

## Heartbeat and stop behavior

rpc_server.py:18 imports functions.running as running. MasterPi.py:20 imports
the same module as Running. functions/running.py:18 defines the function table.
doHeartbeat at running.py:31 sets LastHeartbeat=time.time()+15 and returns a
success tuple. loadFunc calls it at :49. heartbeatTask at :89 checks every 0.1
seconds and calls unloadFunc when the deadline is past and RunningFunc != 0.
The thread starts at import (:102). The loop only catches KeyboardInterrupt;
an exception in unload/exit could kill it. This is a wall-clock function timeout,
not a board command timeout. Heartbeat was NOT called: renewing this deadline
changes when the vendor unloads a function and can prolong its operation.

unloadFunc (:67) calls the active module's exit(), then sets RunningFunc=0.
Remote control is function 1 (:20). remote_control.py:12 and :24 only print and
return from exit/stop, with no board call. Consequently expiry in this mode does
not send a wheel stop. With RunningFunc=0, timeout does nothing at all, including
after direct motor RPCs. No motor-command TTL or keepalive is in the SDK; its
serial timeout at ros_robot_controller_sdk.py:100 is a read timeout.

StopFunc RPC (rpc_server.py:421) queues running.stopFunc. running.py:84 invokes
FUNCTIONS[RunningFunc].stop() without special handling for 0 (None). It is not a
generic emergency stop. Per mode:

- remote_control.py:24: prints only, no stop packet.
- visual_patrol.py:82 and :89: zero chassis velocity on stop/exit.
- avoidance.py:85 and :92: zero velocity, with LED work on exit.
- color_tracking.py:143 and :155: reset, initMove, MotorStop, flags and LEDs.
  initMove (:82) writes a servo and calls arm IK, so stopping can MOVE the arm.
  MotorStop (:88) sends zero chassis velocity.
- color_detect.py:114/:123 and color_sorting.py:122/:131 set flags and RGB.
  Their worker loops at color_detect.py:180 and color_sorting.py:229 react to
  _stop by calling initMove; this can reposition servos rather than freeze them.
- lab_adjust.py:93/:100 toggles run state (stop also resets thresholds).

The zero-velocity path resolves through common/mecanum.py:28, with four duty
writes at :50. Its reset_motors method (:21) also sends four zero duties.
SetBrushMotor at rpc_server.py:110 writes requested duties directly (:127),
without queueing through runbymainth or attaching a motion deadline. SetPWMServo
at :69 writes directly too and converts angle/deviation to pulse at :82.
Therefore the passdown's "every method runs through runbymainth" is too broad.

FinishFunc (rpc_server.py:425) references running.finishFunc, which is absent
from functions/running.py and absent from the full supplied-tree search. As
written, attribute lookup fails before queueing and it sends no board command.
GetRunningFunc (:433) passes the string "GetRunningFunc" to runbymainth, whose
callable check at :385 fails; its following success return is unreachable.
Neither method was called. API behavior on the live image could differ.

## Serial ownership and recommendation, not a deployment decision

MasterPi.py:26 constructs Board; :80 assigns it to RPC handlers. SDK Board opens
/dev/ttyAMA0 at common/ros_robot_controller_sdk.py:100-109 and starts a receiver
at :133. Imports also instantiate Board in common/mecanum.py:6 and
functions/avoidance.py:20. The serial line is therefore not necessarily one
Python object even within MasterPi. Other process ownership remains unmeasured.

Recommend reviewing a replacement driver that owns the serial line exclusively
for the required kill-on-link-loss design. Only killing the process that actually
owns the relevant descriptor can implement that mechanism. This requires an
explicit plan for all vendor controllers/serial owners, verified head calibration,
and preserving/replacing the vendor camera service. No such deployment was made.
The staged SDKBoard interface encodes bounded mappings for review only.

Keeping MasterPi and using a narrow RPC adapter is less disruptive and retains
the camera and vendor UI, but leaves the raw LAN RPC bypass and its weak mode
heartbeat. Killing our RPC proxy leaves MasterPi running with its port open.
StopFunc/FinishFunc are unsuitable substitutes. That alternative needs a separately
reviewed watchdog on the actual serial owner and cannot claim the current
supervisor alone stops motors. Hardware cutoff remains independent of either.

## Port 9026

A whole-tree search for 9026, 9027, socket(), bind(), listen(), SOCK_, TCPServer,
UDPServer and server startup found no 9026/9027 listener in the supplied source.
Known listeners are rpc_server.py:498-501 (9030) and mjpg_server.py:67-69 (8080).
masterpi_pc_software/Arm.py imports socket symbols but contains no matching
second listener. Identity of the observed 9026 process remains UNKNOWN.
No SSH, port-9026 probe, vendor process execution or control-function call was
used for this source analysis.

## Anthony's subsequent decision: dedicated Witness serial

Anthony's direct instruction in this session: "i will not be using the app.
serial is ours only". This selects the exclusive Witness driver architecture.
The earlier recommendation is now selected by Anthony. It does not assert that
live vendor descriptors have already closed. No Pi process was stopped here.

The handover must identify every process holding /dev/ttyAMA0 and the mechanism
that restarts it, stop the vendor controllers and their restart paths, verify
no competing serial owners remain, then construct exactly one Board inside the
supervised Witness driver. Do not use a vendor umbrella module that imports
additional Board objects. Reject startup if another owner exists. The supplied
source has no matching controller service unit or autostart definition, so no
service name or stop command is guessed in this report.

The camera must be split from control before replacing the umbrella process:
MasterPi.py:103-106 launches RPC and MJPEG together; :108-111 opens its camera.
Camera.py:15-80 implements camera capture without constructing a Board, and
mjpg_server.py:67 starts only the HTTP stream. These are reference boundaries
for a camera-only replacement; no vendor file was copied or executed.

Source correction verified after HQ updated the passdown: get_battery at
common/ros_robot_controller_sdk.py:178-190 uses a nonblocking queue read. An
empty queue or disabled reception returns None. Null is not evidence of a dead
serial link. The three receivers can compete, but that cause was not measured
here. The duty-12 floor pulse's lack of motion could reflect stiction or the
control path; the single test did not distinguish them. Do not silently promote
either explanation to a finding.

The serial allocation does not repeat or extend the earlier one-pulse driving
permission. SSH access remains absent on this seat, and no credentials were
provisioned. Live handover has not occurred.
