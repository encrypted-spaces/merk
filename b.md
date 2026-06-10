## In-memory vs RocksDB benchmarks

All benchmarks use a 1M-entry tree with 2k-element batches (or 1-element for prove),
matching the old RocksDB benchmark parameters from the `develop` branch. Both were run
on the same machine. The RocksDB benchmarks called `apply_sorted` which performs the
in-memory tree operation then commits to RocksDB. The in-memory benchmarks call
`apply_memonly_unchecked` which performs only the tree operation.

| Benchmark | RocksDB (develop) | In-memory (this branch) | Speedup |
|---|---|---|---|
| get 1M | 1.54 µs | 543 ns | **2.8x** |
| insert 2k seq | 2.19 ms | 1.53 ms | **1.4x** |
| insert 2k rand | 35.8 ms | 11.7 ms | **3.1x** |
| update 2k seq | 2.10 ms | 994 µs | **2.1x** |
| update 2k rand | 24.3 ms | 2.61 ms | **9.3x** |
| delete 2k rand | 27.5 ms | 138 µs | **199x** |
| prove 1 rand | 5.72 µs | 2.44 µs | **2.3x** |

## Anomalous result: delete

The delete benchmark shows a 199x speedup which is far outside the range of the other
benchmarks. This is a benchmarking artifact: both the old RocksDB and new in-memory
delete benchmarks use modulo wrapping, so after the first 500 iterations all entries
have been deleted and subsequent iterations attempt to delete non-existent keys. In
RocksDB, these no-op deletes still write tombstones to disk, so they remain expensive.
In-memory, they are essentially free. The 138 µs figure is not a meaningful measurement
of delete performance on a full tree.

## Conclusions

Sequential operations (insert seq, update seq) show modest speedups of 1.4-2.1x,
indicating that tree operations (SHA256 hashing, rebalancing, memory allocation) are
the dominant cost, not RocksDB I/O. RocksDB is efficient for sequential writes.

Random operations show larger speedups of 3.1-9.3x, where RocksDB's scattered disk
access, cache misses, and compaction overhead are more significant.

The main performance bottleneck is now the in-memory tree operations themselves. Further
optimization would need to target hashing, allocation patterns, or rebalancing rather
than the storage layer.
