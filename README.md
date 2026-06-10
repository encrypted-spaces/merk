> **Note:** This is a fork of [merk](https://github.com/turbofish-org/merk), maintained as part of the [encrypted-spaces](https://github.com/encrypted-spaces) project. The original README content follows below.

---

<h1 align="left">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./merk-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./merk.svg">
  <img alt="merk" src="./merk.svg">
</picture>
</h1>

*High-performance Merkle key/value store*

Merk is a crypto key/value store - more specifically, it's an in-memory Merkle AVL tree with copy-on-write nodes for efficient snapshots and transactions.

### Features
- **Copy-on-write snapshots** - Cloning a tree root is O(1). Mutations copy only the modified path, so snapshots and writes can coexist cheaply.
- **Transactions** - Clone-mutate-publish with O(1) begin and rollback. Single-writer serialization via a writer mutex.
- **Fast proof generation** - Since Merk implements an AVL tree rather than a trie, it is very efficient to create and verify proofs for ranges of keys.
- **Web-friendly** - Being written in Rust means it is easy to run the proof-verification code in browsers with WebAssembly, allowing for light-clients that can verify data for themselves.

## Usage

**Install:**
```
cargo add merk
```

**Example:**
```rust
use merk::*;

let merk = InMemoryMerk::new();

// apply some operations
let batch = [
    (b"key".to_vec(), Op::Put(b"value".to_vec())),
    (b"key2".to_vec(), Op::Put(b"value2".to_vec())),
    (b"key3".to_vec(), Op::Put(b"value3".to_vec())),
    (b"key4".to_vec(), Op::Delete),
];
merk.apply_sorted(&batch, &[]).unwrap();

// take a snapshot (O(1) — just an Arc refcount bump)
let snap = merk.snapshot();

// further writes don't affect the snapshot
merk.apply_sorted(
    &[(b"key5".to_vec(), Op::Put(b"value5".to_vec()))],
    &[],
).unwrap();

assert!(snap.get(b"key5").unwrap().is_none());
assert_eq!(snap.get(b"key").unwrap(), Some(b"value".to_vec()));
```

## Security

### Security Audits

| Date | Auditor | Scope | Report |
| ---: | :---: | :--- | :---: |
| October 2024 | Trail of Bits | `orga` `merk` `ed` `abci2` | [📄](https://github.com/trailofbits/publications/blob/master/reviews/2024-11-orgaandmerk-securityreview.pdf) |

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use the files in this repository except in compliance with the License. You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

Copyright © 2024 Turbofish, Inc.
