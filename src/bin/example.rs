use asap::{RankingModel, Comparison};
use std::collections::HashMap;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    println!("ASAP: Active Sampling for Pairwise Comparisons");
    println!("==============================================\n");
    
    basic_example();
    
    score_recovery_experiment();
}

fn basic_example() {
    println!("Basic Example:");
    println!("-------------");
    
    let items = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
    let mut model = RankingModel::new(&items);
    
    let comparisons = vec![
        Comparison { winner: "A".to_string(), loser: "B".to_string() },
        Comparison { winner: "B".to_string(), loser: "C".to_string() },
        Comparison { winner: "A".to_string(), loser: "C".to_string() },
        Comparison { winner: "C".to_string(), loser: "D".to_string() },
        Comparison { winner: "B".to_string(), loser: "D".to_string() },
    ];
    
    for comparison in comparisons {
        model.add_comparison(comparison).unwrap();
    }
    
    let scores = model.get_scores().unwrap();
    println!("Inferred scores:");
    for (item, score) in &scores {
        println!("  {}: {:.4}", item, score);
    }
    
    let ranking = model.get_ordering().unwrap();
    println!("\nRanking:");
    for (i, item) in ranking.iter().enumerate() {
        println!("  {}. {}", i + 1, item);
    }
    
    let suggestions = model.suggest_comparisons(3).unwrap();
    println!("\nSuggested comparisons:");
    for (item1, item2) in suggestions {
        println!("  {} vs {}", item1, item2);
    }
    
    let confidence = model.ranking_confidence().unwrap();
    println!("\nRanking confidence: {:.2}", confidence);
    println!();
}

fn score_recovery_experiment() {
    println!("Score Recovery Experiment:");
    println!("-------------------------");
    
    let n_items = 10;
    let n_comparisons = 100;
    let noise_level = 0.2;
    let seed = 42;
    
    let mut rng = StdRng::seed_from_u64(seed);
    let items: Vec<String> = (0..n_items).map(|i| format!("item_{}", i)).collect();
    
    let mut true_scores = HashMap::new();
    for item in &items {
        let score = rng.gen_range(0.0..1.0);
        true_scores.insert(item.clone(), score);
    }
    
    println!("True scores:");
    for (item, score) in &true_scores {
        println!("  {}: {:.4}", item, score);
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
    
    println!("\nGenerated {} comparisons with {:.0}% noise", n_comparisons, noise_level * 100.0);
    
    let mut model = RankingModel::new(&items);
    for comparison in comparisons {
        model.add_comparison(comparison).unwrap();
    }
    
    let inferred_scores = model.get_scores().unwrap();
    
    println!("\nInferred scores:");
    for (item, score) in &inferred_scores {
        println!("  {}: {:.4}", item, score);
    }
    
    let mut true_score_vec = Vec::new();
    let mut inferred_score_vec = Vec::new();
    
    for item in &items {
        true_score_vec.push(*true_scores.get(item).unwrap());
        inferred_score_vec.push(*inferred_scores.get(item).unwrap());
    }
    
    let mut sum_xy = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    
    for i in 0..n_items {
        let x = true_score_vec[i];
        let y = inferred_score_vec[i];
        
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    
    let n = n_items as f64;
    let correlation = (n * sum_xy - sum_x * sum_y) / 
                      ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    println!("\nCorrelation between true and inferred scores: {:.4}", correlation);
    
    let ranking = model.get_ordering().unwrap();
    println!("\nInferred ranking:");
    for (i, item) in ranking.iter().enumerate() {
        let true_score = true_scores.get(item).unwrap();
        println!("  {}. {} (true score: {:.4})", i + 1, item, true_score);
    }
    
    let confidence = model.ranking_confidence().unwrap();
    println!("\nRanking confidence: {:.2}", confidence);
}
