use sha2::{Digest, Sha256};
use std::{convert::TryFrom, num::TryFromIntError};

pub type Hasher = Sha256;
pub const HASH_LENGTH: usize = 32;
pub const NULL_HASH: Hash = [0; HASH_LENGTH];
pub type Hash = [u8; HASH_LENGTH];

// ─── Standard implementations (native / non-zkVM) ───────────────────────────

#[cfg(not(target_os = "zkvm"))]
pub fn kv_hash<D: Digest>(key: &[u8], value: &[u8]) -> Result<Hash, TryFromIntError> {
    let key_length = u32::try_from(key.len())?;
    let val_length = u32::try_from(value.len())?;
    let mut hasher = D::new();
    hasher.update([0]);
    hasher.update(key_length.to_le_bytes());
    hasher.update(key);
    hasher.update(val_length.to_le_bytes());
    hasher.update(value);
    Ok(finalize_to_hash(hasher))
}

#[cfg(not(target_os = "zkvm"))]
pub fn node_hash<D: Digest>(kv: &Hash, left: &Hash, right: &Hash) -> Hash {
    let mut hasher = D::new();
    hasher.update([1]);
    hasher.update(left);
    hasher.update(kv);
    hasher.update(right);
    finalize_to_hash(hasher)
}

#[cfg(not(target_os = "zkvm"))]
fn finalize_to_hash<D: Digest>(hasher: D) -> Hash {
    let res = hasher.finalize();
    let mut hash: Hash = Default::default();
    hash.copy_from_slice(&res[..]);
    hash
}

// ─── RISC0 zkVM implementations ─────────────────────────────────────────────
//
// Calls the SHA-256 accelerator syscall (sys_sha_buffer) directly, bypassing
// the Digest trait wrapper (Sha256::new / update / finalize). This eliminates
// ~50% of per-hash overhead by avoiding hasher object lifecycle, internal
// buffering, and redundant endianness conversions.
//
// Each function assembles its input + FIPS 180-4 padding into an aligned
// buffer and makes a single sys_sha_buffer call:
//
// - node_hash: static pre-padded template (input is always 97 bytes)
// - kv_hash:   MaybeUninit stack buf (variable length, heap fallback for large inputs)
//
// For inputs exceeding the 128-byte stack buffer, a heap-allocated buffer is
// used with the same direct syscall — still faster than the Digest trait.

#[cfg(target_os = "zkvm")]
pub(crate) mod zkvm_sha {
    extern "C" {
        pub fn sys_sha_buffer(
            out_state: *mut [u32; 8],
            in_state: *const [u32; 8],
            buf: *const u8,
            count: u32,
        );
    }

    pub static SHA256_IV: [u32; 8] = [
        0x6a09e667_u32.to_be(),
        0xbb67ae85_u32.to_be(),
        0x3c6ef372_u32.to_be(),
        0xa54ff53a_u32.to_be(),
        0x510e527f_u32.to_be(),
        0x9b05688c_u32.to_be(),
        0x1f83d9ab_u32.to_be(),
        0x5be0cd19_u32.to_be(),
    ];

    /// Compress `blocks` 64-byte blocks from `buf` and return the hash.
    /// The IV is passed by reference (no per-call copy). The output state
    /// is left uninitialized since sys_sha_buffer writes it before reading.
    #[inline(always)]
    pub unsafe fn compress_to_hash(buf: *const u8, blocks: u32) -> [u8; 32] {
        let mut state = core::mem::MaybeUninit::<[u32; 8]>::uninit();
        sys_sha_buffer(state.as_mut_ptr(), &SHA256_IV, buf, blocks);
        // State words are in big-endian format. On LE RISC-V, the native
        // memory layout is already the correct SHA-256 output byte order.
        core::mem::transmute(state.assume_init())
    }

    /// Heap-allocated fallback for inputs that exceed the stack buffer.
    /// Still uses the direct syscall (no Digest trait overhead).
    pub fn hash_bytes(data: &[u8]) -> [u8; 32] {
        let data_len = data.len();
        let padded = (data_len + 9).div_ceil(64) * 64;
        let mut buf = vec![0u8; padded];
        buf[..data_len].copy_from_slice(data);
        buf[data_len] = 0x80;
        buf[padded - 8..padded].copy_from_slice(&((data_len as u64 * 8).to_be_bytes()));
        // Vec is guaranteed to be aligned to at least 4 bytes.
        unsafe { compress_to_hash(buf.as_ptr(), (padded / 64) as u32) }
    }
}

// kv_hash: SHA-256(0x00 || key_len_le(4) || key || val_len_le(4) || value)
#[cfg(target_os = "zkvm")]
pub fn kv_hash<D: Digest>(key: &[u8], value: &[u8]) -> Result<Hash, TryFromIntError> {
    let key_length = u32::try_from(key.len())?;
    let val_length = u32::try_from(value.len())?;
    let key_end = 5 + key.len();
    let data_len = key_end + 4 + value.len();
    Ok(if data_len <= 119 {
        let padded = (data_len + 9).div_ceil(64) * 64;
        #[repr(C, align(4))]
        struct Buf([u8; 128]);
        let mut b = core::mem::MaybeUninit::<Buf>::uninit();
        unsafe {
            let p = b.as_mut_ptr() as *mut u8;
            *p = 0;
            core::ptr::copy_nonoverlapping(key_length.to_le_bytes().as_ptr(), p.add(1), 4);
            core::ptr::copy_nonoverlapping(key.as_ptr(), p.add(5), key.len());
            core::ptr::copy_nonoverlapping(val_length.to_le_bytes().as_ptr(), p.add(key_end), 4);
            core::ptr::copy_nonoverlapping(value.as_ptr(), p.add(key_end + 4), value.len());
            *p.add(data_len) = 0x80;
            core::ptr::write_bytes(p.add(data_len + 1), 0, padded - 8 - (data_len + 1));
            let bit_len = (data_len as u64 * 8).to_be_bytes();
            core::ptr::copy_nonoverlapping(bit_len.as_ptr(), p.add(padded - 8), 8);
            zkvm_sha::compress_to_hash(p, (padded / 64) as u32)
        }
    } else {
        let mut tmp = vec![0u8; data_len];
        tmp[1..5].copy_from_slice(&key_length.to_le_bytes());
        tmp[5..key_end].copy_from_slice(key);
        tmp[key_end..key_end + 4].copy_from_slice(&val_length.to_le_bytes());
        tmp[key_end + 4..].copy_from_slice(value);
        zkvm_sha::hash_bytes(&tmp)
    })
}

// node_hash: SHA-256(0x01 || left(32) || kv(32) || right(32))
// Always 97 bytes → 2 blocks. Static template has tag and padding pre-filled;
// each call only writes the 96 bytes of hash data.
#[cfg(target_os = "zkvm")]
pub fn node_hash<D: Digest>(kv: &Hash, left: &Hash, right: &Hash) -> Hash {
    #[repr(C, align(4))]
    struct Buf([u8; 128]);
    const fn make_template() -> Buf {
        let mut b = [0u8; 128];
        b[0] = 1;
        b[97] = 0x80;
        // 97 * 8 = 776 = 0x0308
        b[126] = 0x03;
        b[127] = 0x08;
        Buf(b)
    }
    static mut TEMPLATE: Buf = make_template();
    unsafe {
        let p = core::ptr::addr_of_mut!(TEMPLATE) as *mut u8;
        core::ptr::copy_nonoverlapping(left.as_ptr(), p.add(1), 32);
        core::ptr::copy_nonoverlapping(kv.as_ptr(), p.add(33), 32);
        core::ptr::copy_nonoverlapping(right.as_ptr(), p.add(65), 32);
        zkvm_sha::compress_to_hash(p, 2)
    }
}

/// Runs hash function correctness checks against known test vectors.
/// Callable from a RISC0 zkVM guest to verify the direct-syscall code paths.
/// Panics on mismatch.
pub fn zkvm_hash_tests() {
    fn hex_to_hash(s: &str) -> Hash {
        let mut h = [0u8; 32];
        for (i, byte) in h.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
        }
        h
    }

    fn check(name: &str, got: Hash, expected: &str) {
        let want = hex_to_hash(expected);
        assert_eq!(got, want, "{name}: hash mismatch");
    }

    // kv_hash: SHA-256(0x00 || key_len_le(4) || key || val_len_le(4) || value)
    check(
        "kv_hash(empty,empty)",
        kv_hash::<Hasher>(&[], &[]).unwrap(),
        "3e7077fd2f66d689e0cee6a7cf5b37bf2dca7c979af356d0a31cbc5c85605c7d",
    );
    check(
        "kv_hash(8xAA,60xBB)",
        kv_hash::<Hasher>(&[0xAA; 8], &[0xBB; 60]).unwrap(),
        "b66736934ae7be5a6716d566bf52b08a130721804f81e028a36c3eab8ed06afb",
    );
    // stack/heap boundary: key(50)+val(60)=110 → data=119 (stack max)
    check(
        "kv_hash(50x07,60x08): stack max",
        kv_hash::<Hasher>(&[7; 50], &[8; 60]).unwrap(),
        "92166bce3af8e3fe490a8ce9cad7034c4fe594646cb325d92f5851662fee6288",
    );
    // key(50)+val(61)=111 → data=120 (heap min)
    check(
        "kv_hash(50x07,61x08): heap min",
        kv_hash::<Hasher>(&[7; 50], &[8; 61]).unwrap(),
        "fc4aa8b17bf27b364a19e4c580b51959d3755eeb021c49fcf527b0ea0e84c897",
    );
    // large: well into heap territory
    check(
        "kv_hash(100x0B,200x0C)",
        kv_hash::<Hasher>(&[0x0B; 100], &[0x0C; 200]).unwrap(),
        "e6285e0227239e8cbc205a6ddc210b9c988a8e6fdfb825092b203f4feced9622",
    );

    // SHA-256 block boundary: data_len=55 → padded=64 (1 block).
    // The zero-fill gap between 0x80 and bit-length is exactly 0 bytes.
    check(
        "kv_hash(46x01,empty): 1-block, zero gap=0",
        kv_hash::<Hasher>(&[0x01; 46], &[]).unwrap(),
        "d540bc25207d2d5d3c4e370cd62699bf162d68197df194ef9e4114616446c911",
    );
    // data_len=56 → padded=128 (2 blocks). First size that crosses block boundary.
    check(
        "kv_hash(47x01,empty): 2-block boundary",
        kv_hash::<Hasher>(&[0x01; 47], &[]).unwrap(),
        "e71cd1198a4068da2d54a267935d9457fd8beedc3a1443bb6669981ddf0bce53",
    );
    // Same boundaries but with empty key, value side
    check(
        "kv_hash(empty,46x02): 1-block",
        kv_hash::<Hasher>(&[], &[0x02; 46]).unwrap(),
        "8fe027855bc16fd4cfcdc5d95cf6abf1b664d0151c50048dfdbd48c2319fe1d8",
    );
    check(
        "kv_hash(empty,47x02): 2-block",
        kv_hash::<Hasher>(&[], &[0x02; 47]).unwrap(),
        "f75e71a75f5bf2005616dd9e8807c49876c43face7fbcad0e6107f70d6172e2c",
    );
    // Minimal non-empty cases
    check(
        "kv_hash(1x42,1x43)",
        kv_hash::<Hasher>(&[0x42], &[0x43]).unwrap(),
        "3ee80cede1b07dcebc0e9de990350713ca9abafb33483dca8fbbeea5ec1d8b24",
    );

    // node_hash: SHA-256(0x01 || left || kv || right)
    let mut kv = [0u8; 32];
    let mut left = [0u8; 32];
    let mut right = [0u8; 32];
    for i in 0..32 {
        kv[i] = ((i * 7 + 3) & 0xFF) as u8;
        left[i] = ((i * 13 + 5) & 0xFF) as u8;
        right[i] = ((i * 17 + 11) & 0xFF) as u8;
    }
    check(
        "node_hash(pattern)",
        node_hash::<Hasher>(&kv, &left, &right),
        "56fd5df21160e080d1b1063c0b8a26dc9380ea101ed06aa1bd3b69003f05b164",
    );
    // All-0xFF: hash data looks like padding bytes — verifies the fixed
    // template positions (tag, 0x80, bit-length) aren't confused by data.
    check(
        "node_hash(all 0xFF)",
        node_hash::<Hasher>(&[0xFF; 32], &[0xFF; 32], &[0xFF; 32]),
        "407c37d0c227a5bf0b227dab96376ae475fb98959e7f6d3cd07e89aa356a59d8",
    );
    // All-zeros: verifies the tag byte (0x01) isn't zeroed by the data writes.
    check(
        "node_hash(all zeros)",
        node_hash::<Hasher>(&[0; 32], &[0; 32], &[0; 32]),
        "d57a85c0063030b53f51d75232d6419f35ac86e5d8807898889463effcf29b7c",
    );
    // Left=0x80 (looks like SHA padding marker), right starts with 0x03,0x08
    // (looks like the bit-length field). Verifies the static template's padding
    // region isn't corrupted by data that mimics padding bytes.
    let mut tricky_kv = [0u8; 32];
    for (i, byte) in tricky_kv.iter_mut().enumerate() {
        *byte = ((i * 3 + 1) & 0xFF) as u8;
    }
    let mut tricky_right = [0u8; 32];
    tricky_right[0] = 0x03;
    tricky_right[1] = 0x08;
    check(
        "node_hash(tricky: 0x80 left, bitlen-like right)",
        node_hash::<Hasher>(&tricky_kv, &[0x80; 32], &tricky_right),
        "376f99477812f636a32e269bdda03327b314b49b8e323b8cd9a74f1508ad31ac",
    );
    // Sequential calls: the static mut TEMPLATE is reused, so verify
    // a second call with different data doesn't retain stale bytes.
    check(
        "node_hash(pattern) again after other calls",
        node_hash::<Hasher>(&kv, &left, &right),
        "56fd5df21160e080d1b1063c0b8a26dc9380ea101ed06aa1bd3b69003f05b164",
    );
}

// ─── Tests ───────────────────────────────────────────────────────────────────
//
// These verify that each hash function produces the correct output by comparing
// against a manual SHA-256 of the assembled input bytes. This validates the
// data layout, padding, and endianness — the same logic used in both the
// Digest-based and zkVM implementations.

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: assemble raw bytes and hash with Digest trait.
    fn sha256_ref(data: &[u8]) -> Hash {
        let mut hasher = Hasher::new();
        hasher.update(data);
        let res = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&res[..]);
        hash
    }

    fn kv_hash_ref(key: &[u8], value: &[u8]) -> Hash {
        let key_length = key.len() as u32;
        let val_length = value.len() as u32;
        let mut data = Vec::new();
        data.push(0);
        data.extend_from_slice(&key_length.to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&val_length.to_le_bytes());
        data.extend_from_slice(value);
        sha256_ref(&data)
    }

    fn node_hash_ref(kv: &Hash, left: &Hash, right: &Hash) -> Hash {
        let mut data = Vec::new();
        data.push(1);
        data.extend_from_slice(left);
        data.extend_from_slice(kv);
        data.extend_from_slice(right);
        sha256_ref(&data)
    }

    #[test]
    fn zkvm_hash_vectors() {
        zkvm_hash_tests();
    }

    // ── kv_hash ──

    fn hex_to_hash(s: &str) -> Hash {
        let mut h = [0u8; 32];
        for (i, byte) in h.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
        }
        h
    }

    #[test]
    fn kv_hash_empty() {
        let h = kv_hash::<Hasher>(&[], &[]).unwrap();
        assert_eq!(h, kv_hash_ref(&[], &[]));
    }

    #[test]
    fn kv_hash_typical() {
        let key = [0xAA; 8];
        let value = vec![0xBB; 60];
        let h = kv_hash::<Hasher>(&key, &value).unwrap();
        assert_eq!(h, kv_hash_ref(&key, &value));
    }

    #[test]
    fn kv_hash_large_value() {
        let key = [0x01; 8];
        let value = vec![0x02; 500];
        let h = kv_hash::<Hasher>(&key, &value).unwrap();
        assert_eq!(h, kv_hash_ref(&key, &value));
    }

    #[test]
    fn kv_hash_large_key_and_value() {
        let key = vec![0x03; 100];
        let value = vec![0x04; 300];
        let h = kv_hash::<Hasher>(&key, &value).unwrap();
        assert_eq!(h, kv_hash_ref(&key, &value));
    }

    // ── kv_hash hardcoded vectors ──
    //
    // Independent of kv_hash_ref — catches layout bugs that affect both
    // the function under test and the reference equally.

    #[test]
    fn kv_hash_vector_empty() {
        let h = kv_hash::<Hasher>(&[], &[]).unwrap();
        assert_eq!(
            h,
            hex_to_hash("3e7077fd2f66d689e0cee6a7cf5b37bf2dca7c979af356d0a31cbc5c85605c7d")
        );
    }

    #[test]
    fn kv_hash_vector_hello_world() {
        let h = kv_hash::<Hasher>(b"hello", b"world").unwrap();
        assert_eq!(
            h,
            hex_to_hash("45d35c0adc3a5ffad47c0a8ecb5c6282b4a5e960b6316ce62d7f095257d7b99a")
        );
    }

    #[test]
    fn kv_hash_vector_typical() {
        let h = kv_hash::<Hasher>(&[0xAA; 8], &[0xBB; 60]).unwrap();
        assert_eq!(
            h,
            hex_to_hash("b66736934ae7be5a6716d566bf52b08a130721804f81e028a36c3eab8ed06afb")
        );
    }

    // ── zkVM stack/heap boundary ──
    //
    // The zkVM kv_hash uses a 128-byte stack buffer when data_len <= 119
    // (key + value <= 110), and falls back to heap allocation above that.

    #[test]
    fn kv_hash_zkvm_boundary_stack_max() {
        // key(50) + value(60) = 110 → data_len = 119 → last stack-buffer size
        let h = kv_hash::<Hasher>(&[7; 50], &[8; 60]).unwrap();
        assert_eq!(h, kv_hash_ref(&[7; 50], &[8; 60]));
        assert_eq!(
            h,
            hex_to_hash("92166bce3af8e3fe490a8ce9cad7034c4fe594646cb325d92f5851662fee6288")
        );
    }

    #[test]
    fn kv_hash_zkvm_boundary_heap_min() {
        // key(50) + value(61) = 111 → data_len = 120 → first heap-fallback size
        let h = kv_hash::<Hasher>(&[7; 50], &[8; 61]).unwrap();
        assert_eq!(h, kv_hash_ref(&[7; 50], &[8; 61]));
        assert_eq!(
            h,
            hex_to_hash("fc4aa8b17bf27b364a19e4c580b51959d3755eeb021c49fcf527b0ea0e84c897")
        );
    }

    // ── node_hash ──

    #[test]
    fn node_hash_zeros() {
        let kv = [0; 32];
        let left = [0; 32];
        let right = [0; 32];
        let h = node_hash::<Hasher>(&kv, &left, &right);
        assert_eq!(h, node_hash_ref(&kv, &left, &right));
    }

    #[test]
    fn node_hash_distinct() {
        let kv = [0x11; 32];
        let left = [0x22; 32];
        let right = [0x33; 32];
        let h = node_hash::<Hasher>(&kv, &left, &right);
        assert_eq!(h, node_hash_ref(&kv, &left, &right));
    }

    #[test]
    fn node_hash_random_pattern() {
        let mut kv = [0u8; 32];
        let mut left = [0u8; 32];
        let mut right = [0u8; 32];
        for i in 0..32 {
            kv[i] = (i * 7 + 3) as u8;
            left[i] = (i * 13 + 5) as u8;
            right[i] = (i * 17 + 11) as u8;
        }
        let h = node_hash::<Hasher>(&kv, &left, &right);
        assert_eq!(h, node_hash_ref(&kv, &left, &right));
    }

    // ── zkVM buffer assembly tests ──
    //
    // The zkVM code paths can't run natively, but the bug-prone part is the
    // buffer assembly (data layout + FIPS 180-4 padding). These tests replicate
    // the exact buffer construction from each zkVM function, hash the result
    // with the Digest trait, and verify it matches the reference output.

    fn sha256_padded_buf(buf: &[u8], data_len: usize) -> Hash {
        let padded = (data_len + 9).div_ceil(64) * 64;
        assert!(buf.len() >= padded);
        assert_eq!(buf[data_len], 0x80);
        for (i, &b) in buf[data_len + 1..padded - 8].iter().enumerate() {
            assert_eq!(b, 0, "non-zero padding at byte {}", data_len + 1 + i);
        }
        let expected_bits = (data_len as u64 * 8).to_be_bytes();
        assert_eq!(&buf[padded - 8..padded], &expected_bits);
        sha256_ref(&buf[..data_len])
    }

    // Replicates the zkVM kv_hash stack buffer assembly (MaybeUninit path)
    fn kv_hash_zkvm_sim(key: &[u8], value: &[u8]) -> Hash {
        let key_length = key.len() as u32;
        let val_length = value.len() as u32;
        let key_end = 5 + key.len();
        let data_len = key_end + 4 + value.len();
        let padded = (data_len + 9).div_ceil(64) * 64;
        if data_len <= 119 {
            let mut buf = [0xFFu8; 128]; // simulate MaybeUninit with non-zero fill
            buf[0] = 0;
            buf[1..5].copy_from_slice(&key_length.to_le_bytes());
            buf[5..key_end].copy_from_slice(key);
            buf[key_end..key_end + 4].copy_from_slice(&val_length.to_le_bytes());
            buf[key_end + 4..data_len].copy_from_slice(value);
            buf[data_len] = 0x80;
            buf[data_len + 1..padded - 8].fill(0);
            buf[padded - 8..padded].copy_from_slice(&((data_len as u64 * 8).to_be_bytes()));
            sha256_padded_buf(&buf, data_len)
        } else {
            let mut tmp = vec![0u8; data_len];
            tmp[1..5].copy_from_slice(&key_length.to_le_bytes());
            tmp[5..key_end].copy_from_slice(key);
            tmp[key_end..key_end + 4].copy_from_slice(&val_length.to_le_bytes());
            tmp[key_end + 4..].copy_from_slice(value);
            sha256_ref(&tmp)
        }
    }

    // Replicates the zkVM node_hash static template assembly
    fn node_hash_zkvm_sim(kv: &Hash, left: &Hash, right: &Hash) -> Hash {
        let mut buf = [0u8; 128];
        buf[0] = 1;
        buf[1..33].copy_from_slice(left);
        buf[33..65].copy_from_slice(kv);
        buf[65..97].copy_from_slice(right);
        buf[97] = 0x80;
        buf[126] = 0x03;
        buf[127] = 0x08;
        sha256_padded_buf(&buf, 97)
    }

    #[test]
    fn zkvm_kv_hash_assembly_sweep() {
        for key_size in 0..=150 {
            for &val_size in &[0usize, 1, 10, 50, 100, 200] {
                let key = vec![(key_size & 0xFF) as u8; key_size];
                let value = vec![(val_size & 0xFF) as u8; val_size];
                let sim = kv_hash_zkvm_sim(&key, &value);
                let reference = kv_hash_ref(&key, &value);
                assert_eq!(
                    sim, reference,
                    "zkvm assembly mismatch at key_size={}, val_size={}",
                    key_size, val_size
                );
            }
        }
    }

    #[test]
    fn zkvm_node_hash_assembly() {
        let mut kv = [0u8; 32];
        let mut left = [0u8; 32];
        let mut right = [0u8; 32];
        for i in 0..32 {
            kv[i] = (i * 7 + 3) as u8;
            left[i] = (i * 13 + 5) as u8;
            right[i] = (i * 17 + 11) as u8;
        }
        let sim = node_hash_zkvm_sim(&kv, &left, &right);
        let reference = node_hash_ref(&kv, &left, &right);
        assert_eq!(sim, reference);
    }

    // ── Sweep tests ──

    #[test]
    fn kv_hash_sweep_sizes() {
        for key_size in 0..=150 {
            for &val_size in &[0usize, 1, 10, 50, 100, 300] {
                let key = vec![(key_size & 0xFF) as u8; key_size];
                let value = vec![(val_size & 0xFF) as u8; val_size];
                let h = kv_hash::<Hasher>(&key, &value).unwrap();
                assert_eq!(
                    h,
                    kv_hash_ref(&key, &value),
                    "mismatch at key_size={}, val_size={}",
                    key_size,
                    val_size
                );
            }
        }
    }
}
