use asap::{Comparison, ItemId, RankingModel};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

fn generate_synthetic_data(
    n_items: usize,
    n_comparisons: usize,
    noise_level: f64,
    seed: u64,
) -> (Vec<ItemId>, Vec<Comparison>, HashMap<ItemId, f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let items: Vec<ItemId> = (0..n_items).map(|i| format!("item_{}", i)).collect();

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
        for j in (i + 1)..n {
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

#[test]
fn test_score_recovery_no_noise() {
    let n_items = 10;
    let n_comparisons = 100;
    let noise_level = 0.0;
    let seed = 42;

    let (items, comparisons, true_scores) =
        generate_synthetic_data(n_items, n_comparisons, noise_level, seed);

    let mut model = RankingModel::new(&items);
    for comparison in comparisons {
        model.add_comparison(comparison).unwrap();
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

    println!("No noise - Kendall's Tau: {}", kendall);
    println!("No noise - Spearman's Rho: {}", spearman);

    assert!(
        kendall > 0.7,
        "Kendall's Tau should be > 0.7, got {}",
        kendall
    );
    assert!(
        spearman > 0.7,
        "Spearman's Rho should be > 0.7, got {}",
        spearman
    );
}

#[test]
fn test_score_recovery_with_noise() {
    let n_items = 10;
    let n_comparisons = 200; // More comparisons to compensate for noise
    let noise_level = 0.2;
    let seed = 42;

    let (items, comparisons, true_scores) =
        generate_synthetic_data(n_items, n_comparisons, noise_level, seed);

    let mut model = RankingModel::new(&items);
    for comparison in comparisons {
        model.add_comparison(comparison).unwrap();
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

    println!("With noise - Kendall's Tau: {}", kendall);
    println!("With noise - Spearman's Rho: {}", spearman);

    assert!(
        kendall > 0.5,
        "Kendall's Tau should be > 0.5, got {}",
        kendall
    );
    assert!(
        spearman > 0.5,
        "Spearman's Rho should be > 0.5, got {}",
        spearman
    );
}

#[test]
fn test_comparison_suggestion() {
    let n_items = 5;
    let n_comparisons = 5; // Few comparisons to test suggestion
    let noise_level = 0.0;
    let seed = 42;

    let (items, comparisons, _) =
        generate_synthetic_data(n_items, n_comparisons, noise_level, seed);

    let mut model = RankingModel::new(&items);
    for comparison in comparisons {
        model.add_comparison(comparison).unwrap();
    }

    let suggested = model.suggest_comparisons(10).unwrap();

    assert!(
        !suggested.is_empty(),
        "Should suggest at least one comparison"
    );

    for (item1, item2) in &suggested {
        assert!(
            items.contains(item1),
            "Suggested item not in item list: {}",
            item1
        );
        assert!(
            items.contains(item2),
            "Suggested item not in item list: {}",
            item2
        );
        assert_ne!(
            item1, item2,
            "Suggested comparison contains same item twice"
        );
    }
}

#[test]
fn test_ranking_confidence() {
    let n_items = 5;
    let seed = 42;

    let items: Vec<ItemId> = (0..n_items).map(|i| format!("item_{}", i)).collect();

    let model = RankingModel::new(&items);
    let confidence1 = model.ranking_confidence().unwrap();

    assert!(
        confidence1 < 0.5,
        "Confidence should be low with no comparisons"
    );

    let mut model = RankingModel::new(&items);

    for i in 0..3 {
        let winner = format!("item_{}", i);
        let loser = format!("item_{}", (i + 1) % n_items);

        model.add_comparison(Comparison { winner, loser }).unwrap();
    }

    let confidence2 = model.ranking_confidence().unwrap();

    assert!(
        confidence2 > confidence1,
        "Confidence should increase with more comparisons"
    );

    for i in 0..n_items {
        for j in 0..n_items {
            if i != j {
                let winner = format!("item_{}", i);
                let loser = format!("item_{}", j);

                let _ = model.add_comparison(Comparison { winner, loser });
            }
        }
    }

    let confidence3 = model.ranking_confidence().unwrap();

    assert!(
        confidence3 > 0.8,
        "Confidence should be high with many comparisons"
    );
}

#[test]
fn test_add_item_and_serialization() {
    let items = vec!["A".to_string(), "B".to_string()];
    let mut model = RankingModel::new(&items);

    model
        .add_comparison(Comparison {
            winner: "A".to_string(),
            loser: "B".to_string(),
        })
        .unwrap();

    model.add_item("C".to_string()).unwrap();

    model
        .add_comparison(Comparison {
            winner: "C".to_string(),
            loser: "A".to_string(),
        })
        .unwrap();
    model
        .add_comparison(Comparison {
            winner: "C".to_string(),
            loser: "B".to_string(),
        })
        .unwrap();

    let scores_before = model.get_scores().unwrap();

    let json = model.to_json().unwrap();

    let mut deserialized_model = RankingModel::from_json(&json).unwrap();

    let scores_after = deserialized_model.get_scores().unwrap();
    for (item, score) in &scores_before {
        assert!(scores_after.contains_key(item));
        assert!((scores_after.get(item).unwrap() - score).abs() < 1e-6);
    }

    deserialized_model.add_item("D".to_string()).unwrap();

    deserialized_model
        .add_comparison(Comparison {
            winner: "D".to_string(),
            loser: "A".to_string(),
        })
        .unwrap();

    let final_scores = deserialized_model.get_scores().unwrap();
    assert!(final_scores.contains_key(&"D".to_string()));

    assert!(
        final_scores.get(&"D".to_string()).unwrap() > final_scores.get(&"A".to_string()).unwrap()
    );
}
