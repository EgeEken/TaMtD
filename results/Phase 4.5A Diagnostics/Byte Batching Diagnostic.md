# Byte batching diagnostic

These are bounded engineering diagnostics, not Phase 4 benchmark results. Each run used the same deterministic 512-sample PBC quality STORE subset for one CUDA training epoch, with no validation/test evaluation and no checkpoint.

| Padded-token cap | Microbatches | Padding reduction vs fixed batches | Examples/s | Peak CUDA allocated / reserved | Peak GPU VRAM | Peak system commit | Minimum available RAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262,144 | 77 | 42.2% | 3.47 | 110 / 224 MiB | 1,096 MiB | 88.8% | 9.9 GiB |
| 1,048,576 | 21 | 40.8% | 11.38 | 379 / 944 MiB | 1,692 MiB | 86.8% | 10.3 GiB |
| 2,097,152 | 17 | 40.0% | 10.77 | 730 / 1,766 MiB | 2,529 MiB | 91.5% | 9.3 GiB |

The 1,048,576-token cap is the default for future token-budget runs: it kept a 40.8% reduction in padded work, had the best measured throughput of the three caps, and used less commit/VRAM than the 2,097,152 cap. The 262,144 cap made microbatches too small for good throughput. The 2,097,152 cap accumulated exactly 32 samples per optimizer update, but its larger batches increased commit and VRAM without improving throughput.

For the same subset, fixed batches of 32 had 43.2% padding, a 2,195,456-token maximum padded batch, and a 17.6 MB maximum int64 token input tensor. The 1,048,576 cap limited batches to 1,048,064 padded tokens and at most 32 samples. Streams were not truncated. Variable microbatch composition and gradient accumulation are recorded in each JSON output; this is not numerically equivalent to the fixed-batch Phase 4 protocol.

The 1,048,576-token diagnostic reached 95% sampled GPU utilization at peak but averaged 16.4% over the short run. CUDA training completed normally. This shows that the bounded condition can execute on the RTX 3070 Ti with the token budget, but the run is not evidence of a sustained speedup over CPU. Existing Phase 4 CPU rows report about 17–22 examples/s; their full training includes validation, test evaluation, and checkpointing, so those values are not a direct timing comparison.

System commit peaked at 86.8% for the 1,048,576 cap and 91.5% for the 2,097,152 cap. No run reached the 95%/98.5% commit stops. The first diagnostic’s minimum C: free space was 31.26 GB; after the child exited, a later system sample showed 38.76 GB free as commit returned toward its pre-run level. The two later runs started after that recovery and kept C: above 38.75 GB. Windows process RSS is unavailable in this telemetry; pagefile.sys size was not measured directly. No existing result row or checkpoint was changed.

Machine-readable details are in `pbc_quality_store_byte_batching.json` (262,144), `pbc_quality_store_byte_batching_1m.json`, and `pbc_quality_store_byte_batching_2m.json`, with runner sidecars alongside them.
