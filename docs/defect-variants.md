# Stratified defect variants

## The problem this solves

Ask for 100 shift defects and, by default, you get 100 independent
random draws. Independent draws clump. A run can end up with 80 small
translations in roughly the same direction and no large rotations at all.
Nothing flags that outcome: the job succeeds, the dataset looks the right size,
and the model trained on it is quietly weak in a corner of the space you never
covered.

Checking coverage only after rendering is the wrong end of the problem:
by then the GPU hours are spent.

`variant_index` replaces the draw with an **assignment**. Sample *n* is told
which region of the parameter space it belongs to, and only samples inside that
region. Coverage stops being a property you measure afterwards and becomes a
property of the run.

## How to use it

Pass the sample's own index. In a loop, that is the loop counter:

```python
for n in range(count):
    job = build_job()
    job.operations.add_shift(count=1, variant_index=n)
    ...
```

Omit it and nothing changes — the legacy uniform sampling is still the default,
and an existing config behaves exactly as before.

The index is not a magnitude, a seed, or an identifier. It is a position in a
repeating sequence: `variant_index=7` and `variant_index=7` on two different
components ask for the same *kind* of defect, not the same numbers. Randomness
still picks the exact values inside the assigned region, so the dataset is not
made of duplicates.

## What each index actually selects

### `add_shift(variant_index=v)`

The shift axes and their strides are as follows:

| What | Cycles over | Advances every |
|---|---|---|
| Mode | `translate`, `rotate`, `combined` | 1 |
| Direction sector | 8 sectors of 45°, in the component's own frame — so "along the pads" and "across the pads" are both reached | 3 |
| Translate magnitude bin | 15–30%, 30–60%, 60–100% of `translate_range` | 24 |
| Rotation magnitude bin | 20–60%, 60–100% of `rotate_z_range` | 72 |
| Rotation sign | `+`, `−` | 144 |

Direction is sampled *inside* the assigned 45° sector, and magnitude *inside*
the assigned bin, so consecutive indices give genuinely different defects that
are nonetheless guaranteed to land in different regions.

The strides form a **mixed-radix odometer**: a digit advances only after every
digit below it has wrapped. That is what makes the axes independent. The full
3 × 8 × 3 × 2 × 2 grid repeats after **288**. Any 288 consecutive indices cover
it exactly one time.

> The independence is not free, and getting it wrong is easy. An earlier
> revision reused strides — rotation bin every 6, rotation sign every 3 — which
> made every direction sector permanently pair with one rotation bin and one
> sign. Sector 0 could never receive a large rotation, and only 72 of the 288
> combinations were reachable. That is exactly the clumping this mechanism
> exists to prevent, so `test_defect_variant_controls.py` now asserts that no
> axis is locked to the sector.

### `add_sideflip(variant_index=v)`

The sideflip axes and their strides are as follows:

| What | Cycles over | Advances every |
|---|---|---|
| Angle bin | 3 equal sub-bins of `[angle_min, angle_max]` | 1 |
| Hinge edge | the two long support edges | 3 |

Repeats after **6**. Without the index, a run can flip every part onto the same
edge and cluster the angles; with it, both edges and the whole angle range are
covered evenly.

## Choosing a count

Because the periods are known, the run length is a decision rather than a guess:

- **shift** — a multiple of **288** covers the full grid. Fewer is fine and
  still evenly spread: mode is covered every 3, direction every 24, translation
  band every 72, rotation band every 144. 288 is the point at which every
  *combination* has appeared one time.
- **sideflip** — a multiple of **6**.

A count that is not a multiple of the period simply leaves the tail partly
covered; it does not skew the earlier samples.

## Two things it is not

- **It is not a seed.** Seeds make a run repeatable; this makes it *spread*.
  They are independent, and you usually want both.
- **It does not make samples identical.** Two samples sharing an index share a
  region, not a value. The values inside the region are still random.

## Verifying the coverage you asked for

Check that your job assigns consecutive variant indices and uses the intended
period and count. The unit tests in
`tests/unittests/core/declarative_api/test_defect_variant_controls.py` verify
the variant-selection rules; they do not validate a rendered dataset.

After execution, check that the requested samples and defects were actually
produced. Failed samples or an insufficient component pool can leave gaps even
when the planned indices cover the full period.

## Related documentation

- Python API: [`cdf/API_SPEC.md`](../cdf/API_SPEC.md) (`variant_index` on shift and sideflip)
- YAML fields: [`pcb_aoi/config/README.md`](../pcb_aoi/config/README.md) (Stratified variants)
