use asap_ranking::{Comparison, RankingModel};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

fn generate_synthetic_data(n_items: usize, seed: u64) -> (Vec<String>, HashMap<String, f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let items: Vec<String> = (0..n_items).map(|i| format!("item_{}", i)).collect();

    let mut true_scores = HashMap::new();
    for item in &items {
        let score = rng.gen_range(0.0..1.0);
        true_scores.insert(item.clone(), score);
    }

    (items, true_scores)
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

    let (items, true_scores) = generate_synthetic_data(n_items, seed);

    let mut model = RankingModel::<String>::new(&items);
    let mut rng_comparisons = ChaCha8Rng::seed_from_u64(seed); // For simulating comparison outcomes

    for _ in 0..n_comparisons {
        let suggestions = model.suggest_comparisons(1).unwrap();
        if suggestions.is_empty() {
            break; // No more suggestions
        }
        let (suggested_item1, suggested_item2) = suggestions[0].clone();

        // Simulate outcome
        let score1 = true_scores.get(&suggested_item1).unwrap();
        let score2 = true_scores.get(&suggested_item2).unwrap();

        let prob_item1_wins = 1.0 / (1.0 + (-10.0f64 * (score1 - score2)).exp());
        // noise_level is 0.0 for this test
        let is_noisy_comparison = rng_comparisons.gen_range(0.0..1.0) < noise_level;

        let item1_actually_wins = if is_noisy_comparison {
            rng_comparisons.gen_range(0.0..1.0) < 0.5
        } else {
            rng_comparisons.gen_range(0.0..1.0) < prob_item1_wins
        };

        let comparison = if item1_actually_wins {
            Comparison::<String> {
                winner: suggested_item1.clone(),
                loser: suggested_item2.clone(),
            }
        } else {
            Comparison::<String> {
                winner: suggested_item2.clone(),
                loser: suggested_item1.clone(),
            }
        };
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

    let (items, true_scores) = generate_synthetic_data(n_items, seed);

    let mut model = RankingModel::<String>::new(&items);
    let mut rng_comparisons = ChaCha8Rng::seed_from_u64(seed); // For simulating comparison outcomes

    for _ in 0..n_comparisons {
        let suggestions = model.suggest_comparisons(1).unwrap();
        if suggestions.is_empty() {
            break; // No more suggestions
        }
        let (suggested_item1, suggested_item2) = suggestions[0].clone();

        // Simulate outcome
        let score1 = true_scores.get(&suggested_item1).unwrap();
        let score2 = true_scores.get(&suggested_item2).unwrap();

        let prob_item1_wins = 1.0 / (1.0 + (-10.0f64 * (score1 - score2)).exp());
        let is_noisy_comparison = rng_comparisons.gen_range(0.0..1.0) < noise_level;

        let item1_actually_wins = if is_noisy_comparison {
            rng_comparisons.gen_range(0.0..1.0) < 0.5
        } else {
            rng_comparisons.gen_range(0.0..1.0) < prob_item1_wins
        };

        let comparison = if item1_actually_wins {
            Comparison::<String> {
                winner: suggested_item1.clone(),
                loser: suggested_item2.clone(),
            }
        } else {
            Comparison::<String> {
                winner: suggested_item2.clone(),
                loser: suggested_item1.clone(),
            }
        };
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
    let n_initial_comparisons = 5; // Few comparisons to build an initial model
    let noise_level = 0.0; // For simulating outcomes during model build-up
    let seed = 42;

    let (items, true_scores) = generate_synthetic_data(n_items, seed);

    let mut model = RankingModel::<String>::new(&items);
    let mut rng_comparisons = ChaCha8Rng::seed_from_u64(seed);

    for _ in 0..n_initial_comparisons {
        let suggestions = model.suggest_comparisons(1).unwrap();
        if suggestions.is_empty() {
            break;
        }
        let (suggested_item1, suggested_item2) = suggestions[0].clone();

        let score1 = true_scores.get(&suggested_item1).unwrap();
        let score2 = true_scores.get(&suggested_item2).unwrap();

        let prob_item1_wins = 1.0 / (1.0 + (-10.0f64 * (score1 - score2)).exp());
        let is_noisy_comparison = rng_comparisons.gen_range(0.0..1.0) < noise_level;

        let item1_actually_wins = if is_noisy_comparison {
            rng_comparisons.gen_range(0.0..1.0) < 0.5
        } else {
            rng_comparisons.gen_range(0.0..1.0) < prob_item1_wins
        };

        let comparison_to_add = if item1_actually_wins {
            Comparison::<String> {
                winner: suggested_item1.clone(),
                loser: suggested_item2.clone(),
            }
        } else {
            Comparison::<String> {
                winner: suggested_item2.clone(),
                loser: suggested_item1.clone(),
            }
        };
        model.add_comparison(comparison_to_add).unwrap();
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

    let items: Vec<String> = (0..n_items).map(|i| format!("item_{}", i)).collect();

    let model = RankingModel::<String>::new(&items);
    let confidence1 = model.ranking_confidence().unwrap();

    assert!(
        confidence1 < 0.5,
        "Confidence should be low with no comparisons"
    );

    // Add enough comparisons that each item has >= 5 comparisons
    let mut model = RankingModel::<String>::new(&items);
    for _ in 0..3 {
        for i in 0..n_items {
            let winner = format!("item_{}", i);
            let loser = format!("item_{}", (i + 1) % n_items);
            model
                .add_comparison(Comparison::<String> { winner, loser })
                .unwrap();
        }
    }

    // Compute scores so confidence uses the discrimination component
    let _ = model.get_scores().unwrap();
    let confidence2 = model.ranking_confidence().unwrap();

    assert!(
        confidence2 > confidence1,
        "Confidence should increase with comparisons and computed scores"
    );

    // Add many more comparisons with a clear ordering (item_0 > item_1 > ... > item_4)
    for _ in 0..3 {
        for i in 0..n_items {
            for j in (i + 1)..n_items {
                let winner = format!("item_{}", i);
                let loser = format!("item_{}", j);
                let _ = model.add_comparison(Comparison::<String> { winner, loser });
            }
        }
    }

    let _ = model.get_scores().unwrap();
    let confidence3 = model.ranking_confidence().unwrap();

    assert!(
        confidence3 > confidence2,
        "Confidence should increase with more consistent comparisons: {confidence2} -> {confidence3}"
    );
}

#[test]
fn test_add_item_and_serialization() {
    let items = vec!["A".to_string(), "B".to_string()];
    let mut model = RankingModel::<String>::new(&items);

    model
        .add_comparison(Comparison::<String> {
            winner: "A".to_string(),
            loser: "B".to_string(),
        })
        .unwrap();

    model.add_item("C".to_string()).unwrap();

    model
        .add_comparison(Comparison::<String> {
            winner: "C".to_string(),
            loser: "A".to_string(),
        })
        .unwrap();
    model
        .add_comparison(Comparison::<String> {
            winner: "C".to_string(),
            loser: "B".to_string(),
        })
        .unwrap();

    let _scores_before_serialization = model.get_scores().unwrap();
    assert_eq!(model.data.item_count(), 3);

    // Test removing an item before serialization
    model.remove_item(&"B".to_string()).unwrap();
    assert_eq!(model.data.item_count(), 2);
    assert!(model.scores.is_none()); // Scores are cleared after removal

    let scores_after_removal_before_serialization = model.get_scores().unwrap();
    assert!(scores_after_removal_before_serialization.contains_key(&"A".to_string()));
    assert!(scores_after_removal_before_serialization.contains_key(&"C".to_string()));
    assert!(!scores_after_removal_before_serialization.contains_key(&"B".to_string()));

    let json = model.to_json().unwrap();
    let mut deserialized_model = RankingModel::<String>::from_json(&json).unwrap();

    assert_eq!(deserialized_model.data.item_count(), 2);
    let scores_after_deserialization = deserialized_model.get_scores().unwrap();

    for (item, score) in &scores_after_removal_before_serialization {
        assert!(scores_after_deserialization.contains_key(item));
        assert!((scores_after_deserialization.get(item).unwrap() - score).abs() < 1e-6);
    }
    assert!(!scores_after_deserialization.contains_key(&"B".to_string()));

    // Test adding and removing after deserialization
    deserialized_model.add_item("D".to_string()).unwrap();
    assert_eq!(deserialized_model.data.item_count(), 3);
    deserialized_model
        .add_comparison(Comparison::<String> {
            winner: "D".to_string(),
            loser: "A".to_string(),
        })
        .unwrap();

    let scores_before_another_removal = deserialized_model.get_scores().unwrap();
    assert!(scores_before_another_removal.contains_key(&"D".to_string()));
    assert!(
        scores_before_another_removal.get(&"D".to_string()).unwrap()
            > scores_before_another_removal.get(&"A".to_string()).unwrap()
    );

    deserialized_model.remove_item(&"A".to_string()).unwrap();
    assert_eq!(deserialized_model.data.item_count(), 2);
    let final_scores = deserialized_model.get_scores().unwrap();
    assert!(final_scores.contains_key(&"D".to_string()));
    assert!(final_scores.contains_key(&"C".to_string()));
    assert!(!final_scores.contains_key(&"A".to_string()));
}
