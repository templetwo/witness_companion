# Codex, gpt-6-astra, Jetson seat, 2026-10-12 EDT
import unittest

from companion.arm_lift import forward, plan_lift, plan_reach, _inverse, validate_pulses

START = {3: 754, 4: 2485, 5: 845, 6: 1405}


class ArmLiftTests(unittest.TestCase):
    def test_observed_start_matches_vendor_upright_pose(self):
        pose = forward(START)
        self.assertAlmostEqual(pose['x_cm'], 0, places=6)
        self.assertAlmostEqual(pose['y_cm'], 6, delta=.01)
        self.assertAlmostEqual(pose['z_cm'], 18, delta=.01)
        self.assertAlmostEqual(pose['pitch_degrees'], 0, places=6)
        self.assertEqual(_inverse(pose), START)

    def test_one_cm_lift_and_reverse_stay_in_sampled_corridor(self):
        plan = plan_lift(START)
        self.assertEqual(plan['target'], {3: 701, 4: 2335, 5: 943, 6: 1405})
        self.assertAlmostEqual(plan['predicted_end']['z_cm'] - plan['predicted_start']['z_cm'], 1, delta=.02)
        self.assertEqual(plan['samples'][0]['pulses'], START)
        self.assertEqual(plan['samples'][-1]['pulses'], plan['target'])
        for sample in reversed(plan['samples']):
            validate_pulses(sample['pulses'])
        self.assertFalse(plan['physical_position_verified'])

    def test_offsets_checked_after_conversion_not_clipped(self):
        # This target fits the vendor's nominal pulse range, but the elbow
        # calibration offset would put its final pulse beyond 2500us.
        with self.assertRaisesRegex(ValueError, 'offset pulse'):
            _inverse({'x_cm': 0, 'y_cm': 7, 'z_cm': 18, 'pitch_degrees': 0})

    def test_invalid_and_uncommissioned_inputs_rejected(self):
        for height in [0, -.5, 1.1, float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                plan_lift(START, height)
        for pulses in [{**START, 2: 1500}, {**START, 3: True},
                       {**START, 3: 500}, {**START, 6: 1600}, {**START, 4: 2600}]:
            with self.assertRaises(ValueError):
                plan_lift(pulses)

    def test_unreachable_and_wrong_half_plane_rejected(self):
        for y, z in [(6, 100), (-6, 18), (0, 18), (6, float('nan'))]:
            with self.assertRaises(ValueError):
                _inverse({'x_cm': 0, 'y_cm': y, 'z_cm': z, 'pitch_degrees': 0})

    def test_shorter_lifts_round_trip_through_geometry(self):
        for distance in [.25, .5, .75]:
            plan = plan_lift(START, distance)
            self.assertEqual(_inverse(forward(plan['target'])), plan['target'])

    def test_reach_has_bounded_stages_and_exact_reverse(self):
        plan = plan_reach(START)
        self.assertEqual(plan['target'], {3: 986, 4: 2288, 5: 1273, 6: 1405})
        self.assertAlmostEqual(plan['predicted_end']['y_cm'] - plan['predicted_start']['y_cm'], 5, delta=.02)
        self.assertAlmostEqual(plan['predicted_end']['z_cm'] - plan['predicted_start']['z_cm'], 2, delta=.02)
        self.assertEqual(plan['return_waypoints'], list(reversed([START] + plan['outbound_waypoints'][:-1])))
        previous = START
        for target in plan['outbound_waypoints'] + plan['return_waypoints']:
            validate_pulses(target)
            self.assertLessEqual(max(abs(target[c] - previous[c]) for c in START), 150)
            self.assertEqual(target[6], START[6])
            previous = target
        self.assertEqual(previous, START)

    def test_reach_rejects_uncommissioned_region_and_base_rotation(self):
        for start in [{**START, 6: 1500}, {**START, 3: 1700}]:
            with self.assertRaises(ValueError):
                plan_reach(start)
