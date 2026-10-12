# Motor cutoff: the decision reduced to what needs Anthony's word

HQ, 2026-10-12. gate-shape:physical-safety. Prior decisions: the house has no
prior ruling of this shape (searched memory, the Stack chronicle domains
witness-companion*, and the lane runs 2026-10-11/12). The evidence it rests on:
kill test B on 2026-10-11 (driver frozen with the serial port open, wheels kept
turning the full 10 s until a second connection stopped them) and Codex's
2026-10-12 note that a frozen Pi or kernel still has no stopping guarantee.

## What is already covered without hardware
The Temple guard kills its driver on a missed Jetson heartbeat, a missed driver
response, EOF, link failure or supervisor death, and the kill closes the serial
port, which the 10-11 kill test A showed stops the wheels within about a second.
Covered: a hung driver, a dropped Jetson, a crashed guard. Not covered: a frozen
Pi kernel, a wedged serial driver, a Pi that loses power to its USB while the
motor rail stays up. Those are the cases a cutoff is for.

## The fork
Option A: a physical switch in the battery lead, hand operated. Cost under ten
dollars, zero software, works when everything is frozen, but only while a hand is
within reach. This is the attended-only answer and it is what you have been doing
with the pack switch.

Option B: a normally-open relay in the motor power lead held closed by a watchdog
output that the Jetson toggles (GPIO on the Jetson, or a cheap hardware watchdog
timer module that opens the relay if it is not pulsed every N seconds). If the
Jetson or its heartbeat stops, the motors lose power. Cost about twenty dollars,
one evening of wiring, and it makes unattended floor time defensible. The
Jetson's own freeze then needs a second guard, which is why the timer module
beats a bare GPIO.

Option C: no cutoff, attended only, written down as the rule. Honest, free, and
it means the robot never runs without a hand on the switch.

## HQ's recommendation
A now (it is what you already do; put it in the rule), B before any unattended
floor time, and C is not a decision, it is the current state. The thing only you
can decide is whether unattended floor time is a goal at all this season. If it
is not, A plus the written rule closes this item today.

## What HQ needs from you
One line: "A and the rule", or "build B", or "not this season".
