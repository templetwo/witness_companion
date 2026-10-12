# Gripper mechanical check (power off, Anthony's hands)

HQ, 2026-10-12. Reason: during the 17:18 to 17:37 grasp attempt the jaws
contacted and shifted the green block at 1700, 1600 and 1500 us closure but did
not hold it, and Anthony reported the arm quiet and the grip loose. A quiet
servo with a loose grip is more often a slipped horn or a loose linkage than a
weak servo. Ten minutes with the power off settles it before any more closure
attempts.

Battery switch OFF, USB power off, Jetson power bank disconnected from the Pi
(or Pi shut down). Then:

1. Horn screw. The gripper servo output has a plastic or metal horn held by one
   centre screw. Hold the servo body and try to rotate the horn by hand: it should
   not move at all. Any play means the screw is loose or the spline is stripped.
2. Linkage pins. Follow the horn to the jaw linkage. Each pin or screw should be
   seated; the links should not lift off their posts.
3. Jaw play. Hold the jaws closed by hand and feel for free play in the pivot.
   Compare with how firmly the jaw pads meet; pads that touch at one corner only
   will push a block out instead of holding it.
4. Pads. Rubber pads present on both jaws, clean, not glazed.
5. Range by hand. With power off, gently open and close the jaws through their
   travel. Note where they bind, if anywhere, and the fully-closed gap.
6. Block fit. Put one block between the open jaws and close by hand. If the jaws
   cannot close square on the block, the approach height, not the servo, is the
   problem.

Record the six answers in two or three lines in the session receipt. If 1 or 2
show play: fix it, then run the gripper batch of CALIBRATION_SESSION.md before
anything else. If all six are clean: the grip problem is pulse or placement, and
the calibration session's gripper batch (gap versus pulse) is the next step.
