# HQ attended grasp session, 2026-10-12 18:4x to 19:0x EDT: result

Authorization: Anthony at the terminal, "ok let me see you grab the block and lift it", "go", then
"im not going to rearange, that was a cost paid, now use the ultrasonic and other cammereas to
reorient and pic the green block up". Anthony at the switch throughout.

**Result: no pickup.** Nine batches through `scripts/attended/hq_grasp_once.py` (receipts in
`receipts/hq-grasp-20261012/`, frames in `jetson:~/witness_scratch/hq-grasp-20261012/`, hashes in
`INDEX.json` there and here).

| Batch | Move | What the cameras showed |
|---|---|---|
| dry0 | none | pose matched start, both cameras fresh |
| seat | shoulder 2016 -> 2216, jaws 1800 | pad tips landed on the green block's top edge and scattered the stack; stop rule |
| clear1 | shoulder back to 2016 | arm clear |
| open2000 | jaws 1800 -> 2000 | first time the open jaws were wider than the block |
| yaw1 | base 1520 -> 1620 | jaws swung toward the side camera; fork view slid right |
| yaw2 | base 1620 -> 1320 | wrong way; blocks left the fork view |
| survey | base 1520, shoulder 1916 | wider fork view; green still out of frame |
| approach1 | base 1420, shoulder 2016 -> 2216 | pads above the green block, block centred in the fork view, 3 cm too high |
| approach2 | elbow 1803 -> 1603, shoulder 2216 -> 2416 | elbow sign wrong: gripper swung down and back to the blue block; stop rule |
| retreat | reverse, then shoulder up | three of four stages; battery floor tripped at 6984 mV on the last |

End state: jaws open 2000, clear above the blocks, commanded 1=2000, 3=782, 4=1803, 5=2116,
6=1420 (not home); guard locked and disarmed, override removed, normal ExecStart. Blocks on the
floor: red, green directly behind it, blue apart. Charge before any motion.

What was learned, as measured signs: base yaw + moves the jaws toward the side camera; shoulder
+100 us lowers the pads about 2 cm; elbow decreasing swings the gripper down and back; gripper 2000
clears the block. Next attempt from the end pose: yaw 1420, shoulder to 2216, elbow increasing with
shoulder to reach floor level outward over green, close 2000 -> 1500, lift. About four batches on a
fresh charge.
