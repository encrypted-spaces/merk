use crate::avl::node::Node;
use crate::test_utils::*;
use crate::*;
use rand::prelude::*;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::iter::FromIterator;

const ITERATIONS: usize = 2_000;
type Map = BTreeMap<Vec<u8>, Vec<u8>>;

#[test]
fn fuzz() {
    let mut rng = thread_rng();

    for _ in 0..ITERATIONS {
        let seed = rng.gen::<u64>();
        fuzz_case(seed);
    }
}

#[test]
fn fuzz_17391518417409062786() {
    fuzz_case(17391518417409062786);
}

#[test]
fn fuzz_396148930387069749() {
    fuzz_case(396148930387069749);
}

fn fuzz_case(seed: u64) {
    let mut rng: SmallRng = SeedableRng::seed_from_u64(seed);
    let initial_size = (rng.gen::<u64>() % 10) + 1;
    let tree = make_tree_rand(initial_size, initial_size, seed);
    let mut map = Map::from_iter(tree.iter());
    let mut maybe_tree = Some(tree);
    println!("====== MERK FUZZ ======");
    println!("SEED: {}", seed);
    println!("{:?}", maybe_tree.as_ref().unwrap());

    for j in 0..3 {
        let batch_size = (rng.gen::<u64>() % 3) + 1;
        let batch = make_batch(maybe_tree.as_ref(), batch_size, rng.gen::<u64>());
        println!("BATCH {}", j);
        println!("{:?}", batch);
        maybe_tree = apply_to_memonly(maybe_tree, &batch);
        apply_to_map(&mut map, &batch);
        assert_map(maybe_tree.as_ref(), &map);
        if let Some(tree) = &maybe_tree {
            println!("{:?}", &tree);
        } else {
            println!("(Empty tree)");
        }
    }
}

fn make_batch(maybe_tree: Option<&Node>, size: u64, seed: u64) -> Vec<BatchEntry> {
    let rng: RefCell<SmallRng> = RefCell::new(SeedableRng::seed_from_u64(seed));
    let mut batch = Vec::with_capacity(size as usize);

    let get_random_key = || {
        let tree = maybe_tree.as_ref().unwrap();
        let entries: Vec<_> = tree.iter().collect();
        let index = rng.borrow_mut().gen::<u64>() as usize % entries.len();
        entries[index].0.clone()
    };

    let random_value = |size| {
        let mut value = vec![0; size];
        rng.borrow_mut().fill_bytes(&mut value[..]);
        value
    };

    let insert = || (random_value(2), Op::Put(random_value(2)));
    let update = || {
        let key = get_random_key();
        (key.to_vec(), Op::Put(random_value(2)))
    };
    let delete = || {
        let key = get_random_key();
        (key.to_vec(), Op::Delete)
    };

    for _ in 0..size {
        let entry = if maybe_tree.is_some() {
            let kind = rng.borrow_mut().gen::<u64>() % 3;
            if kind == 0 {
                insert()
            } else if kind == 1 {
                update()
            } else {
                delete()
            }
        } else {
            insert()
        };
        batch.push(entry);
    }
    batch.sort_by(|a, b| a.0.cmp(&b.0));

    // remove dupes
    let mut maybe_prev_key: Option<Vec<u8>> = None;
    let mut deduped_batch = Vec::with_capacity(batch.len());
    for entry in batch {
        if let Some(prev_key) = &maybe_prev_key {
            if *prev_key == entry.0 {
                continue;
            }
        }

        maybe_prev_key = Some(entry.0.clone());
        deduped_batch.push(entry);
    }
    deduped_batch
}

fn apply_to_map(map: &mut Map, batch: &Batch) {
    for entry in batch.iter() {
        match entry {
            (key, Op::Put(value)) => {
                map.insert(key.to_vec(), value.to_vec());
            }
            (key, Op::Delete) => {
                map.remove(key);
            }
            (start, Op::DeleteRange(end)) => {
                let keys_to_remove: Vec<Vec<u8>> = map
                    .range(start.to_vec()..end.to_vec())
                    .map(|(k, _)| k.clone())
                    .collect();
                for k in keys_to_remove {
                    map.remove(&k);
                }
            }
        }
    }
}

#[test]
fn fuzz_in_place_parity() {
    let mut rng = thread_rng();

    for _ in 0..ITERATIONS {
        let seed = rng.gen::<u64>();
        fuzz_in_place_case(seed);
    }
}

fn fuzz_in_place_case(seed: u64) {
    use crate::avl::walker::Walker;
    use crate::avl::PanicSource;

    let mut rng: SmallRng = SeedableRng::seed_from_u64(seed);
    let initial_size = (rng.gen::<u64>() % 10) + 1;
    let tree = make_tree_rand(initial_size, initial_size, seed);
    let mut maybe_tree_func = Some(tree.clone());
    let mut maybe_tree_ip = Some(tree);

    for _ in 0..3 {
        let batch_size = (rng.gen::<u64>() % 3) + 1;
        let batch = make_batch(maybe_tree_func.as_ref(), batch_size, rng.gen::<u64>());

        maybe_tree_func = apply_to_memonly(maybe_tree_func, &batch);

        match &maybe_tree_ip {
            Some(tree_ip) => {
                let mut walker = Walker::new(tree_ip.clone(), PanicSource {});
                let mut batch_mut = batch.to_vec();
                match walker.apply_in_place(&mut batch_mut) {
                    Ok(_deleted_keys) => {
                        let mut t = walker.into_inner();
                        t.commit();
                        assert_tree_invariants(&t);

                        match &maybe_tree_func {
                            Some(func) => {
                                assert_eq!(func.hash(), t.hash(), "hash mismatch at seed {}", seed,);
                            }
                            None => {
                                panic!(
                                    "functional path returned None but in-place succeeded (seed {})",
                                    seed,
                                );
                            }
                        }
                        maybe_tree_ip = Some(t);
                    }
                    Err(_) => {
                        assert!(
                            maybe_tree_func.is_none(),
                            "in-place errored but functional path returned Some (seed {})",
                            seed,
                        );
                        return;
                    }
                }
            }
            None => return,
        }
    }
}

fn assert_map(maybe_tree: Option<&Node>, map: &Map) {
    if map.is_empty() {
        assert!(maybe_tree.is_none(), "expected tree to be None");
        return;
    }

    let tree = maybe_tree.expect("expected tree to be Some");

    let expected_iter = map.iter();
    let tree_iter = tree.iter();
    for (tree_kv, map_kv) in tree_iter.zip(expected_iter) {
        assert_eq!(tree_kv.0, *map_kv.0);
        assert_eq!(tree_kv.1, *map_kv.1);
    }

    assert_eq!(tree.iter().count(), map.len());
}
