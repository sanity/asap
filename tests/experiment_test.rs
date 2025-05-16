use asap::{Comparison, RankingModel};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::time::Instant;

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

fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    let n = a.len();
    if n <= 1 {
        return 1.0; // Perfect correlation for 0 or 1 item
    }

    let mut concordant = 0;
    let mut discordant = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let a_i = a[i];
            let a_j = a[j];
            let b_i = b[i];
            let b_j = b[j];

            if a_i.is_nan() || a_j.is_nan() || b_i.is_nan() || b_j.is_nan() {
                continue; // Skip NaN comparisons
            }

            let a_order = match a_i.partial_cmp(&a_j) {
                Some(order) => order,
                None => continue, // Skip incomparable values
            };

            let b_order = match b_i.partial_cmp(&b_j) {
                Some(order) => order,
                None => continue, // Skip incomparable values
            };

            if a_order == b_order {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let total_pairs = concordant + discordant;
    if total_pairs == 0 {
        return 0.0; // No valid comparisons
    }

    (concordant as f64 - discordant as f64) / (total_pairs as f64)
}

fn spearman_correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    let n = a.len();
    if n <= 1 {
        return 1.0; // Perfect correlation for 0 or 1 item
    }

    let mut valid_indices = Vec::new();
    for i in 0..n {
        if !a[i].is_nan() && !b[i].is_nan() {
            valid_indices.push(i);
        }
    }

    let valid_n = valid_indices.len();
    if valid_n <= 1 {
        return 1.0; // Perfect correlation for 0 or 1 valid item
    }

    let mut a_ranks = vec![0.0; valid_n];
    let mut b_ranks = vec![0.0; valid_n];

    for (idx, &i) in valid_indices.iter().enumerate() {
        let mut a_rank = 1;
        let mut b_rank = 1;

        for &j in &valid_indices {
            if i == j {
                continue;
            }

            if let Some(ordering) = a[j].partial_cmp(&a[i]) {
                if ordering == std::cmp::Ordering::Less {
                    a_rank += 1;
                }
            }

            if let Some(ordering) = b[j].partial_cmp(&b[i]) {
                if ordering == std::cmp::Ordering::Less {
                    b_rank += 1;
                }
            }
        }

        a_ranks[idx] = a_rank as f64;
        b_ranks[idx] = b_rank as f64;
    }

    let mut sum_d_squared = 0.0;
    for i in 0..valid_n {
        let d = a_ranks[i] - b_ranks[i];
        sum_d_squared += d * d;
    }

    if valid_n < 2 {
        return 0.0; // Cannot compute correlation with less than 2 valid items
    }

    1.0 - (6.0 * sum_d_squared) / (valid_n as f64 * ((valid_n * valid_n) - 1) as f64)
}

fn run_experiment(n_items: usize, noise_level: f64, seed: u64) -> Vec<(usize, f64, f64, f64)> {
    let max_comparisons = n_items * 10; // Scale comparisons with item count
    let step_size = max_comparisons / 10; // Measure at 10 points

    let (items, all_comparisons, true_scores) =
        generate_synthetic_data(n_items, max_comparisons, noise_level, seed);

    let mut results = Vec::new();

    for n_comparisons in (step_size..=max_comparisons).step_by(step_size) {
        let start_time = Instant::now();

        let mut model = RankingModel::<String>::new(&items);
        for i in 0..n_comparisons {
            if i < all_comparisons.len() {
                model.add_comparison(all_comparisons[i].clone()).unwrap();
            }
        }

        let inferred_scores = model.get_scores().unwrap();

        let mut true_score_vec = Vec::new();
        let mut inferred_score_vec = Vec::new();

        for item in &items {
            true_score_vec.push(*true_scores.get(item).unwrap());
            inferred_score_vec.push(*inferred_scores.get(item).unwrap());
        }

        let kendall = kendall_tau(&true_score_vec, &inferred_score_vec);
        let spearman = spearman_correlation(&true_score_vec, &inferred_score_vec);

        let elapsed = start_time.elapsed().as_secs_f64();

        results.push((n_comparisons, kendall, spearman, elapsed));
    }

    results
}

#[test]
fn test_experiment_with_5_items() {
    let n_items = 5;
    let noise_level = 0.1;
    let seed = 42;

    let results = run_experiment(n_items, noise_level, seed);

    println!("\nExperiment with {} items:", n_items);
    println!("Comparisons | Kendall's Tau | Spearman's Rho | Time (s)");
    println!("---------------------------------------------------------");

    for (n_comparisons, kendall, spearman, time) in &results {
        println!(
            "{:11} | {:12.4} | {:13.4} | {:8.4}",
            n_comparisons, kendall, spearman, time
        );
    }

    let (_, final_kendall, final_spearman, _) = results.last().unwrap();
    assert!(*final_kendall > 0.6, "Final Kendall's Tau should be > 0.6");
    assert!(
        *final_spearman > 0.6,
        "Final Spearman's Rho should be > 0.6"
    );
}

#[test]
fn test_experiment_with_50_items() {
    let n_items = 50;
    let noise_level = 0.1;
    let seed = 42;

    let results = run_experiment(n_items, noise_level, seed);

    println!("\nExperiment with {} items:", n_items);
    println!("Comparisons | Kendall's Tau | Spearman's Rho | Time (s)");
    println!("---------------------------------------------------------");

    for (n_comparisons, kendall, spearman, time) in &results {
        println!(
            "{:11} | {:12.4} | {:13.4} | {:8.4}",
            n_comparisons, kendall, spearman, time
        );
    }

    let (_, final_kendall, final_spearman, _) = results.last().unwrap();
    assert!(*final_kendall > 0.5, "Final Kendall's Tau should be > 0.5");
    assert!(
        *final_spearman > 0.5,
        "Final Spearman's Rho should be > 0.5"
    );
}

#[test]
fn test_experiment_with_200_items() {
    let n_items = 200;
    let noise_level = 0.1;
    let seed = 42;

    let results = run_experiment(n_items, noise_level, seed);

    println!("\nExperiment with {} items:", n_items);
    println!("Comparisons | Kendall's Tau | Spearman's Rho | Time (s)");
    println!("---------------------------------------------------------");

    for (n_comparisons, kendall, spearman, time) in &results {
        println!(
            "{:11} | {:12.4} | {:13.4} | {:8.4}",
            n_comparisons, kendall, spearman, time
        );
    }

    let (_, final_kendall, final_spearman, _) = results.last().unwrap();
    assert!(*final_kendall > 0.4, "Final Kendall's Tau should be > 0.4");
    assert!(
        *final_spearman > 0.4,
        "Final Spearman's Rho should be > 0.4"
    );
}
