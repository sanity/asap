use asap::{Comparison, RankingModel};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

fn generate_synthetic_data(
    n_items: usize,
    n_comparisons: usize,
    noise_level: f64,
    seed: u64,
) -> (Vec<String>, Vec<Comparison<String>>, HashMap<String, f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let items: Vec<String> = (0..n_items).map(|i| format!("item_{}", i)).collect();

    let mut true_scores = HashMap::new();
    for item in &items {
        let score = rng.gen_range(0.0..1.0);
        true_scores.insert(item.clone(), score);
    }

    let mut comparisons = Vec::new();
    for _ in 0..n_comparisons {
        let idx1 = rng.gen_range(0..n_items);
        let mut idx2 = rng.gen_range(0..n_items);
        while idx2 == idx1 {
            idx2 = rng.gen_range(0..n_items);
        }

        let item1 = &items[idx1];
        let item2 = &items[idx2];

        let score1 = true_scores.get(item1).unwrap();
        let score2 = true_scores.get(item2).unwrap();

        let prob_item1_wins = 1.0 / (1.0 + (-10.0f64 * (score1 - score2)).exp());
        let noise = rng.gen_range(0.0..1.0) < noise_level;

        let item1_wins = if noise {
            rng.gen_range(0.0..1.0) < 0.5 // Random choice if noisy
        } else {
            rng.gen_range(0.0..1.0) < prob_item1_wins // Probabilistic choice based on scores
        };

        let comparison = if item1_wins {
            Comparison::<String> {
                winner: item1.clone(),
                loser: item2.clone(),
            }
        } else {
            Comparison::<String> {
                winner: item2.clone(),
                loser: item1.clone(),
            }
        };

        comparisons.push(comparison);
    }

    (items, comparisons, true_scores)
}

fn bench_accurate_algorithm(c: &mut Criterion) {
    let (items, comparisons, _) = generate_synthetic_data(20, 100, 0.1, 42);

    c.bench_function("accurate_algorithm", |b| {
        b.iter(|| {
            let mut model = RankingModel::<String>::new_with_options(&items, false, false);
            for comparison in &comparisons {
                model.add_comparison(comparison.clone()).unwrap();
            }
            black_box(model.get_scores().unwrap());
        })
    });
}

fn bench_approximate_algorithm(c: &mut Criterion) {
    let (items, comparisons, _) = generate_synthetic_data(20, 100, 0.1, 42);

    c.bench_function("approximate_algorithm", |b| {
        b.iter(|| {
            let mut model = RankingModel::<String>::new_with_options(&items, true, false);
            for comparison in &comparisons {
                model.add_comparison(comparison.clone()).unwrap();
            }
            black_box(model.get_scores().unwrap());
        })
    });
}

fn bench_suggest_comparisons(c: &mut Criterion) {
    let (items, comparisons, _) = generate_synthetic_data(20, 100, 0.1, 42);

    c.bench_function("suggest_comparisons", |b| {
        let mut model = RankingModel::<String>::new(&items);
        for comparison in &comparisons {
            model.add_comparison(comparison.clone()).unwrap();
        }

        b.iter(|| {
            black_box(model.suggest_comparisons(5).unwrap());
        })
    });
}

criterion_group!(
    benches,
    bench_accurate_algorithm,
    bench_approximate_algorithm,
    bench_suggest_comparisons
);
criterion_main!(benches);
