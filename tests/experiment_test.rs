use asap::{RankingModel, Comparison, ItemId};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::time::Instant;

fn generate_synthetic_data(
    n_items: usize,
    n_comparisons: usize,
    noise_level: f64,
    seed: u64,
) -> (Vec<ItemId>, Vec<Comparison>, HashMap<ItemId, f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    
    let items: Vec<ItemId> = (0..n_items)
        .map(|i| format!("item_{}", i))
        .collect();
    
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
            Comparison {
                winner: item1.clone(),
                loser: item2.clone(),
            }
        } else {
            Comparison {
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
    let mut concordant = 0;
    let mut discordant = 0;
    
    for i in 0..n {
        for j in (i+1)..n {
            let a_order = a[i].partial_cmp(&a[j]).unwrap();
            let b_order = b[i].partial_cmp(&b[j]).unwrap();
            
            if a_order == b_order {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    
    let total_pairs = (n * (n - 1)) / 2;
    (concordant as f64 - discordant as f64) / (total_pairs as f64)
}

fn spearman_correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    
    let n = a.len();
    
    let mut a_ranks = Vec::with_capacity(n);
    let mut b_ranks = Vec::with_capacity(n);
    
    for i in 0..n {
        let mut a_rank = 1;
        let mut b_rank = 1;
        
        for j in 0..n {
            if i == j {
                continue;
            }
            
            if a[j] < a[i] {
                a_rank += 1;
            }
            
            if b[j] < b[i] {
                b_rank += 1;
            }
        }
        
        a_ranks.push(a_rank as f64);
        b_ranks.push(b_rank as f64);
    }
    
    let mut sum_d_squared = 0.0;
    for i in 0..n {
        let d = a_ranks[i] - b_ranks[i];
        sum_d_squared += d * d;
    }
    
    1.0 - (6.0 * sum_d_squared) / (n as f64 * ((n * n) - 1) as f64)
}

fn run_experiment(n_items: usize, noise_level: f64, seed: u64) -> Vec<(usize, f64, f64, f64)> {
    let max_comparisons = n_items * 10; // Scale comparisons with item count
    let step_size = max_comparisons / 10; // Measure at 10 points
    
    let (items, all_comparisons, true_scores) = generate_synthetic_data(
        n_items, max_comparisons, noise_level, seed
    );
    
    let mut results = Vec::new();
    
    for n_comparisons in (step_size..=max_comparisons).step_by(step_size) {
        let start_time = Instant::now();
        
        let mut model = RankingModel::new(&items);
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
    
    for (n_comparisons, kendall, spearman, time) in results {
        println!("{:11} | {:12.4} | {:13.4} | {:8.4}", 
                 n_comparisons, kendall, spearman, time);
    }
    
    let (_, final_kendall, final_spearman, _) = results.last().unwrap();
    assert!(*final_kendall > 0.6, "Final Kendall's Tau should be > 0.6");
    assert!(*final_spearman > 0.6, "Final Spearman's Rho should be > 0.6");
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
    
    for (n_comparisons, kendall, spearman, time) in results {
        println!("{:11} | {:12.4} | {:13.4} | {:8.4}", 
                 n_comparisons, kendall, spearman, time);
    }
    
    let (_, final_kendall, final_spearman, _) = results.last().unwrap();
    assert!(*final_kendall > 0.5, "Final Kendall's Tau should be > 0.5");
    assert!(*final_spearman > 0.5, "Final Spearman's Rho should be > 0.5");
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
    
    for (n_comparisons, kendall, spearman, time) in results {
        println!("{:11} | {:12.4} | {:13.4} | {:8.4}", 
                 n_comparisons, kendall, spearman, time);
    }
    
    let (_, final_kendall, final_spearman, _) = results.last().unwrap();
    assert!(*final_kendall > 0.4, "Final Kendall's Tau should be > 0.4");
    assert!(*final_spearman > 0.4, "Final Spearman's Rho should be > 0.4");
}
