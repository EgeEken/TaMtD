# PBC attribution representations

The diagnostic cache is derived from the existing PBC quality STORE files. It does not re-encode the source images.

- `pbc_init_only`: the original image header and the first three full-image patches, one for each channel.
- `pbc_residual_only`: a canonical header with zero base values followed by the original residual patch records. It is a classification representation and is not treated as a normal reconstruction benchmark.
- `pbc_init_zeroed`: the complete original stream with every initialization patch's serialized grid index replaced by zero. The initialization mask, palette bounds, cell size, and all field widths are retained, so this is a parsed-field counterfactual rather than arbitrary byte zeroing.
- `pbc_patch_shuffled`: the original header and initialization patches followed by a deterministic reordering of complete residual patch records. Individual patch fields are never split. Decoded pixels were checked for equality against the original stream.

The current format has no explicit initialization-patch count field. For these 32x32 RGB CIFAR samples, the encoder emits one initial full-image patch per channel, so the diagnostic parser uses `channels` (three) as the initialization count.
