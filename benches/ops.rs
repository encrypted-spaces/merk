use criterion::{criterion_group, criterion_main, Criterion};
use merk::proofs::query::QueryItem;
use merk::test_utils::*;

fn get_1m_memonly(c: &mut Criterion) {
    let initial_size = 1_000_000;
    let batch_size = 2_000;
    let num_batches = initial_size / batch_size;

    let mut batches = vec![];
    for i in 0..num_batches {
        batches.push(make_batch_rand(batch_size, i));
    }

    let tree = make_tree_rand(initial_size, batch_size, 0);

    let mut i = 0;
    c.bench_function("get_1m_memonly", |b| {
        b.iter(|| {
            let batch_index = (i % num_batches) as usize;
            let key_index = (i / num_batches) as usize;

            let key = &batches[batch_index][key_index].0;
            tree.get(key);

            i = (i + 1) % initial_size;
        })
    });
}

fn prove_1m_1_rand_memonly(c: &mut Criterion) {
    let initial_size = 1_000_000;
    let batch_size = 1_000;

    let tree = make_tree_rand(initial_size, batch_size, 0);

    let mut i = 0;
    c.bench_function("prove_1m_1_rand_memonly", |b| {
        b.iter(|| {
            let batch = make_batch_rand(1, i);
            let keys: Vec<QueryItem> = batch
                .into_iter()
                .map(|(key, _)| QueryItem::Key(key))
                .collect();
            tree.prove(keys).expect("prove failed");
            i = (i + 1) % (initial_size / batch_size);
        })
    });
}

fn insert_1m_2k_seq_memonly(c: &mut Criterion) {
    let initial_size = 1_000_000;
    let batch_size = 2_000;

    let mut tree = Some(make_tree_seq(initial_size));

    let mut i = initial_size / batch_size;
    c.bench_function("insert_1m_2k_seq_memonly", |b| {
        b.iter(|| {
            let batch = make_batch_seq((i * batch_size)..((i + 1) * batch_size));
            tree = Some(apply_memonly_unchecked(tree.take().unwrap(), &batch));
            i += 1;
        })
    });
}

fn insert_1m_2k_rand_memonly(c: &mut Criterion) {
    let initial_size = 1_000_000;
    let batch_size = 2_000;

    let mut tree = Some(make_tree_rand(initial_size, batch_size, 0));

    let mut i = initial_size / batch_size;
    c.bench_function("insert_1m_2k_rand_memonly", |b| {
        b.iter(|| {
            let batch = make_batch_rand(batch_size, i);
            tree = Some(apply_memonly_unchecked(tree.take().unwrap(), &batch));
            i += 1;
        })
    });
}

fn update_1m_2k_seq_memonly(c: &mut Criterion) {
    let initial_size = 1_000_000;
    let batch_size = 2_000;

    let mut tree = Some(make_tree_seq(initial_size));

    let mut i = 0;
    c.bench_function("update_1m_2k_seq_memonly", |b| {
        b.iter(|| {
            let batch = make_batch_seq((i * batch_size)..((i + 1) * batch_size));
            tree = Some(apply_memonly_unchecked(tree.take().unwrap(), &batch));
            i = (i + 1) % (initial_size / batch_size);
        })
    });
}

fn update_1m_2k_rand_memonly(c: &mut Criterion) {
    let initial_size = 1_000_000;
    let batch_size = 2_000;

    let mut tree = Some(make_tree_rand(initial_size, batch_size, 0));

    let mut i = 0;
    c.bench_function("update_1m_2k_rand_memonly", |b| {
        b.iter(|| {
            let batch = make_batch_rand(batch_size, i);
            tree = Some(apply_memonly_unchecked(tree.take().unwrap(), &batch));
            i = (i + 1) % (initial_size / batch_size);
        })
    });
}

fn delete_1m_2k_rand_memonly(c: &mut Criterion) {
    let initial_size = 1_000_000;
    let batch_size = 2_000;

    let mut tree = Some(make_tree_rand(initial_size, batch_size, 0));

    let mut i = 0;
    c.bench_function("delete_1m_2k_rand_memonly", |b| {
        b.iter(|| {
            let batch = make_del_batch_rand(batch_size, i);
            tree = Some(apply_memonly_unchecked(tree.take().unwrap(), &batch));
            i = (i + 1) % (initial_size / batch_size);
        })
    });
}

criterion_group!(
    benches,
    get_1m_memonly,
    insert_1m_2k_seq_memonly,
    insert_1m_2k_rand_memonly,
    update_1m_2k_seq_memonly,
    update_1m_2k_rand_memonly,
    delete_1m_2k_rand_memonly,
    prove_1m_1_rand_memonly,
);
criterion_main!(benches);
