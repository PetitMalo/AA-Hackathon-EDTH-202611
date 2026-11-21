# Terrain scenes

Each scene is one JSON file, `<scene>.json`, with a fixed layout and a list
of target box poses. Load it with `TerrainAviary(terrain="field@v1")`, replacing
`field` with `forest`, `village`, or `city`, and `v1` with `v<N>` to use the
N-th pose in `targets`.

- `targets`: the 4 × 1.65 × 1.55 m [`target_box.urdf`](../objects/target_box.urdf)
  poses, `position_m` at ground level and `rpy_rad`, with a `context` label
  that describes the hiding spot.
- `objects`: the static props (`type: urdf`, from [`objects/`](../objects/README.md)),
  then the ground and road surfaces (`type: surface`, `size_m` and `rgba`). The first
  surface is the 500 × 500 m ground, centred at (0, 0).

`load_terrain` returns the box body first, then the objects in file order. The
detector tracks only the box.

| Scene | Layout | Bodies | Targets |
| --- | --- | ---: | ---: |
| `field` | Crop parcels split by hedgerows, dirt tracks with a power line, two farmsteads with sheds, hay bales, and a paddock | 256 | 18 |
| `forest` | Pines in the east and broadleaves in the west, denser near the centre, with undergrowth, boulders, clearings, trails, and a hut | 378 | 40 |
| `village` | A main road with two side streets, houses facing the road with driveways, cars, fences, hedges, sheds, and gardens; farms and fields outside | 435 | 40 |
| `city` | A street grid with perimeter blocks, alleys, courtyards, two parks, a car park, kerbside parking, and a suburban yard outside | 346 | 40 |

Props stop 150 m from the centre (in x or y), except within 20 m of a box pose, because PyBullet's
CPU renderer draws every triangle, however far away; the ground surfaces still
cover the whole tile. The start at `(0, 0, 3)` is clear: nothing stands within
9 m of the origin.

No prop except a bush touches any box pose, and any two poses are at least 12 m
apart. Poses `v1`–`v6` are hand-placed, each with cover built around it:

| Scene | v1 | v2 | v3 | v4 | v5 | v6 |
| --- | --- | --- | --- | --- | --- | --- |
| `field` | Hedgerow, under an oak | Among hay bales | Behind a shed | Paddock corner, under an oak | Rock pile | Track junction, under an oak |
| `forest` | Undergrowth, under an oak | Clearing edge, under an oak | Trailside bushes | Between boulders | Behind a hut | Inside a pine stand |
| `village` | Backyard by a shed | Alley between houses | Driveway, in the car's place | Garden, under an oak | Rear fence corner | Copse outside the village |
| `city` | Block courtyard | Alley between buildings | Car park slot | Kerbside among parked cars | Park undergrowth | Suburban yard, by a shed |

The generator picks the remaining poses from candidates placed beside props,
such as walls, parked cars, tree canopies, bushes, rocks, and hay bales. It
renders each candidate in PyBullet and keeps the best-hidden ones, measured by
the share of the box visible from straight above and by how many of 8 low views
from 25 m away see it. It takes the contexts in turn so that no single kind of
hiding spot dominates. The open field yields fewer well-hidden spots than the
other scenes.

`previews/<scene>.png` shows each layout from above, with numbered rings at
the poses; the [overview](previews/contact_sheet.png) shows all four.
`previews/<scene>_targets.png` shows an oblique close view of every pose, labelled
with its context and visibility.
[`tools/generate_assets.py`](../../../tools/generate_assets.py) regenerates the
scenes, poses, and previews from fixed seeds. `village@v1` is the default for
`ApproachAviary` and the RC script.

## Attack maps

The `level_*` terrains train the attack phase, once the target has been located.
Each file uses a `setups` list instead of `targets`: every setup pairs a ground
target pose (`target`, as above) with a drone `start` pose (`position_m`,
`rpy_rad`), chosen so the target begins inside the onboard camera's FOV at a
fixed range. `load_terrain` places the box at `setups[v-1].target`, and
`terrain_start_pose("<name>@v<N>")` returns the matching start; `ApproachAviary`
reads it when no explicit `start_pose` is passed.

`level_1` is an empty 500 × 500 m map (only the ground surface). Sixty setups,
ten per distance, the distance doubling from 5 m to 160 m:

| Versions | Distance |
| --- | ---: |
| v1–v10 | 5 m |
| v11–v20 | 10 m |
| v21–v30 | 20 m |
| v31–v40 | 40 m |
| v41–v50 | 80 m |
| v51–v60 | 160 m |

Each setup randomises the target's position and orientation in the tile and the
drone's azimuth and height (2–30 m, capped per distance so the target stays in
the 70° VFOV within the 60° pitch limit), and a start attitude that keeps the
target inside the 102° HFOV. Regenerate with
`uv run python tools/generate_assets.py --level-1`; `previews/level_1.png`
shows every drone–target pair from above, and `level_1_targets.png` a thumbnail
per setup.
