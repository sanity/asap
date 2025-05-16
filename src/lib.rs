
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;
use statrs::distribution::ContinuousCDF;

pub type ItemId = String;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comparison {
    pub winner: ItemId,
    pub loser: ItemId,
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} > {}", self.winner, self.loser)
    }
}

#[derive(Error, Debug)]
pub enum AsapError {
    #[error("Item not found: {0}")]
    ItemNotFound(ItemId),
    #[error("Invalid comparison: both items are the same")]
    InvalidComparison,
    #[error("Not enough comparisons to compute scores")]
    NotEnoughComparisons,
    #[error("Internal algorithm error: {0}")]
    InternalError(String),
}

#[derive(Debug, Clone)]
pub struct ComparisonMatrix {
    item_indices: HashMap<ItemId, usize>,
    index_to_item: Vec<ItemId>,
    win_counts: Vec<Vec<usize>>,
    comparison_count: usize,
}

impl ComparisonMatrix {
    pub fn new(items: &[ItemId]) -> Self {
        let n = items.len();
        let mut item_indices = HashMap::with_capacity(n);
        let mut index_to_item = Vec::with_capacity(n);
        
        for (idx, item) in items.iter().enumerate() {
            item_indices.insert(item.clone(), idx);
            index_to_item.push(item.clone());
        }
        
        let win_counts = vec![vec![0; n]; n];
        
        ComparisonMatrix {
            item_indices,
            index_to_item,
            win_counts,
            comparison_count: 0,
        }
    }
    
    pub fn add_comparison(&mut self, comparison: &Comparison) -> Result<(), AsapError> {
        if comparison.winner == comparison.loser {
            return Err(AsapError::InvalidComparison);
        }
        
        let winner_idx = self.item_indices.get(&comparison.winner)
            .ok_or_else(|| AsapError::ItemNotFound(comparison.winner.clone()))?;
        let loser_idx = self.item_indices.get(&comparison.loser)
            .ok_or_else(|| AsapError::ItemNotFound(comparison.loser.clone()))?;
        
        self.win_counts[*winner_idx][*loser_idx] += 1;
        self.comparison_count += 1;
        
        Ok(())
    }
    
    pub fn get_win_count(&self, item_i: &ItemId, item_j: &ItemId) -> Result<usize, AsapError> {
        let i_idx = self.item_indices.get(item_i)
            .ok_or_else(|| AsapError::ItemNotFound(item_i.clone()))?;
        let j_idx = self.item_indices.get(item_j)
            .ok_or_else(|| AsapError::ItemNotFound(item_j.clone()))?;
        
        Ok(self.win_counts[*i_idx][*j_idx])
    }
    
    pub fn get_comparison_count(&self, item_i: &ItemId, item_j: &ItemId) -> Result<usize, AsapError> {
        let i_idx = self.item_indices.get(item_i)
            .ok_or_else(|| AsapError::ItemNotFound(item_i.clone()))?;
        let j_idx = self.item_indices.get(item_j)
            .ok_or_else(|| AsapError::ItemNotFound(item_j.clone()))?;
        
        Ok(self.win_counts[*i_idx][*j_idx] + self.win_counts[*j_idx][*i_idx])
    }
    
    pub fn item_count(&self) -> usize {
        self.index_to_item.len()
    }
    
    pub fn total_comparisons(&self) -> usize {
        self.comparison_count
    }
    
    pub fn items(&self) -> Vec<ItemId> {
        self.index_to_item.clone()
    }
    
    pub fn get_item_index(&self, item: &ItemId) -> Result<usize, AsapError> {
        self.item_indices.get(item)
            .copied()
            .ok_or_else(|| AsapError::ItemNotFound(item.clone()))
    }
    
    pub fn get_item_from_index(&self, index: usize) -> Option<ItemId> {
        self.index_to_item.get(index).cloned()
    }
}

pub struct RankingModel {
    pub data: ComparisonMatrix,
    pub scores: Option<HashMap<ItemId, f64>>,
    approximate: bool,
    selective_eig: bool,
}

impl RankingModel {
    pub fn new(items: &[ItemId]) -> Self {
        RankingModel {
            data: ComparisonMatrix::new(items),
            scores: None,
            approximate: false,
            selective_eig: false,
        }
    }
    
    pub fn new_with_options(items: &[ItemId], approximate: bool, selective_eig: bool) -> Self {
        RankingModel {
            data: ComparisonMatrix::new(items),
            scores: None,
            approximate,
            selective_eig,
        }
    }

    pub fn add_comparison(&mut self, comparison: Comparison) -> Result<(), AsapError> {
        self.data.add_comparison(&comparison)?;
        self.scores = None;
        Ok(())
    }

    pub fn get_ordering(&mut self) -> Result<Vec<ItemId>, AsapError> {
        let scores = self.get_scores()?;
        
        let mut items_with_scores: Vec<_> = scores.iter().collect();
        
        items_with_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(items_with_scores.into_iter().map(|(item, _)| item.clone()).collect())
    }

    pub fn get_scores(&mut self) -> Result<HashMap<ItemId, f64>, AsapError> {
        if let Some(ref scores) = self.scores {
            return Ok(scores.clone());
        }
        
        let scores = if self.approximate {
            self.compute_approximate_scores()?
        } else {
            self.compute_accurate_scores()?
        };
        
        self.scores = Some(scores.clone());
        Ok(scores)
    }

    pub fn suggest_comparisons(&self, max: usize) -> Result<Vec<(ItemId, ItemId)>, AsapError> {
        if self.data.item_count() < 2 {
            return Err(AsapError::NotEnoughComparisons);
        }
        
        let mut pairs_with_gain = Vec::new();
        let items = self.data.items();
        let n = items.len();
        
        let scores = if let Some(ref scores) = self.scores {
            scores.clone()
        } else {
            let mut temp_scores = HashMap::new();
            for i in 0..n {
                let item_i = self.data.get_item_from_index(i)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
                
                let mut wins = 0;
                let mut total = 0;
                
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    
                    let item_j = self.data.get_item_from_index(j)
                        .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
                    
                    wins += self.data.get_win_count(&item_i, &item_j)?;
                    total += self.data.get_comparison_count(&item_i, &item_j)?;
                }
                
                let score = if total > 0 { wins as f64 / total as f64 } else { 0.5 };
                temp_scores.insert(item_i, score);
            }
            temp_scores
        };
        
        for i in 0..n {
            for j in (i+1)..n {
                let item_i = &items[i];
                let item_j = &items[j];
                
                let score_i = *scores.get(item_i).unwrap_or(&0.5);
                let score_j = *scores.get(item_j).unwrap_or(&0.5);
                
                let prob_i_wins = 1.0 / (1.0 + (-10.0 * (score_i - score_j)).exp());
                
                let info_gain = -(prob_i_wins * prob_i_wins.ln() + (1.0 - prob_i_wins) * (1.0 - prob_i_wins).ln());
                
                let comparison_count = self.data.get_comparison_count(item_i, item_j)?;
                let adjusted_gain = info_gain / (1.0 + 0.1 * comparison_count as f64);
                
                pairs_with_gain.push((adjusted_gain, (item_i.clone(), item_j.clone())));
            }
        }
        
        pairs_with_gain.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let result = pairs_with_gain.into_iter()
            .take(max)
            .map(|(_, pair)| pair)
            .collect();
        
        Ok(result)
    }

    pub fn ranking_confidence(&self) -> Result<f64, AsapError> {
        let n = self.data.item_count();
        if n <= 1 {
            return Ok(1.0); // Only one item, so we're 100% confident
        }
        
        let scores = if let Some(ref scores) = self.scores {
            scores.clone()
        } else {
            let max_comparisons = (n * (n - 1)) / 2;
            let confidence = (self.data.total_comparisons() as f64) / (max_comparisons as f64);
            return Ok(1.0 / (1.0 + (-5.0 * (confidence - 0.5)).exp()));
        };
        
        let mut score_vec = Vec::with_capacity(n);
        let items = self.data.items();
        
        for item in &items {
            score_vec.push(*scores.get(item).unwrap_or(&0.5));
        }
        
        let mut sorted_scores = score_vec.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut avg_diff = 0.0;
        let mut count = 0;
        
        for i in 0..(sorted_scores.len() - 1) {
            let diff = (sorted_scores[i] - sorted_scores[i + 1]).abs();
            avg_diff += diff;
            count += 1;
        }
        
        if count == 0 {
            return Ok(0.5); // Default confidence if no differences
        }
        
        avg_diff /= count as f64;
        
        let mut variance = 0.0;
        let mean = score_vec.iter().sum::<f64>() / n as f64;
        
        for score in &score_vec {
            variance += (score - mean) * (score - mean);
        }
        
        variance /= n as f64;
        
        
        let comparison_factor = (self.data.total_comparisons() as f64) / ((n * (n - 1)) / 2) as f64;
        let diff_factor = avg_diff / 0.1; // Normalize by expected difference
        let variance_factor = variance / 0.1; // Normalize by expected variance
        
        let raw_confidence = 0.4 * comparison_factor + 0.3 * diff_factor + 0.3 * variance_factor;
        
        let confidence = 1.0 / (1.0 + (-5.0 * (raw_confidence - 0.5)).exp());
        
        Ok(confidence)
    }

    pub fn is_sufficiently_confident(&self, threshold: f64) -> Result<bool, AsapError> {
        let confidence = self.ranking_confidence()?;
        Ok(confidence >= threshold)
    }
    
    
    fn compute_accurate_scores(&self) -> Result<HashMap<ItemId, f64>, AsapError> {
        use nalgebra::DVector;
        use statrs::distribution::{Normal, Continuous};
        
        let n = self.data.item_count();
        let mut scores = HashMap::new();
        
        if n == 0 {
            return Ok(scores);
        }
        
        let mut mu = DVector::zeros(n);
        let mut sigma = DVector::from_element(n, 1.0);
        
        let beta = 0.1f64; // Skill variability
        let tau = 0.05f64; // Dynamic factor
        
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                
                let item_i = self.data.get_item_from_index(i)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
                let item_j = self.data.get_item_from_index(j)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
                
                let wins_i_over_j = self.data.get_win_count(&item_i, &item_j)?;
                let wins_j_over_i = self.data.get_win_count(&item_j, &item_i)?;
                
                if wins_i_over_j == 0 && wins_j_over_i == 0 {
                    continue;
                }
                
                for _ in 0..wins_i_over_j {
                    let v = (2.0 * beta * beta + sigma[i] * sigma[i] + sigma[j] * sigma[j]).sqrt();
                    let mean_diff = mu[i] - mu[j];
                    
                    let normal = Normal::new(0.0, 1.0).unwrap();
                    let c = v * normal.pdf(mean_diff / v) / normal.cdf(mean_diff / v);
                    
                    mu[i] = mu[i] + sigma[i] * sigma[i] * c / v;
                    mu[j] = mu[j] - sigma[j] * sigma[j] * c / v;
                    
                    let factor: f64 = 1.0 - sigma[i] * sigma[i] * sigma[j] * sigma[j] * c * (c + mean_diff / v) / (v * v);
                    sigma[i] = sigma[i] * factor.sqrt();
                    sigma[j] = sigma[j] * factor.sqrt();
                }
                
                for _ in 0..wins_j_over_i {
                    let v = (2.0 * beta * beta + sigma[i] * sigma[i] + sigma[j] * sigma[j]).sqrt();
                    let mean_diff = mu[j] - mu[i];
                    
                    let normal = Normal::new(0.0, 1.0).unwrap();
                    let c = v * normal.pdf(mean_diff / v) / normal.cdf(mean_diff / v);
                    
                    mu[j] = mu[j] + sigma[j] * sigma[j] * c / v;
                    mu[i] = mu[i] - sigma[i] * sigma[i] * c / v;
                    
                    let factor: f64 = 1.0 - sigma[i] * sigma[i] * sigma[j] * sigma[j] * c * (c + mean_diff / v) / (v * v);
                    sigma[i] = sigma[i] * factor.sqrt();
                    sigma[j] = sigma[j] * factor.sqrt();
                }
            }
        }
        
        for i in 0..n {
            let item = self.data.get_item_from_index(i)
                .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
            scores.insert(item, mu[i]);
        }
        
        Ok(scores)
    }
    
    fn compute_approximate_scores(&self) -> Result<HashMap<ItemId, f64>, AsapError> {
        use nalgebra::DVector;
        use statrs::distribution::{Normal, Continuous};
        use rand::prelude::*;
        
        let n = self.data.item_count();
        let mut scores = HashMap::new();
        
        if n == 0 {
            return Ok(scores);
        }
        
        let mut mu = DVector::zeros(n);
        let mut sigma = DVector::from_element(n, 1.0);
        
        let beta = 0.1f64; // Skill variability
        let tau = 0.05f64; // Dynamic factor
        
        let mut all_comparisons = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                
                let item_i = self.data.get_item_from_index(i)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
                let item_j = self.data.get_item_from_index(j)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
                
                let wins_i_over_j = self.data.get_win_count(&item_i, &item_j)?;
                
                for _ in 0..wins_i_over_j {
                    all_comparisons.push((i, j));
                }
            }
        }
        
        let mut rng = rand::thread_rng();
        all_comparisons.shuffle(&mut rng);
        
        for (winner_idx, loser_idx) in all_comparisons {
            let v = (2.0 * beta * beta + sigma[winner_idx] * sigma[winner_idx] + sigma[loser_idx] * sigma[loser_idx]).sqrt();
            
            let mean_diff = mu[winner_idx] - mu[loser_idx];
            
            let normal = Normal::new(0.0, 1.0).unwrap();
            let c = v * normal.pdf(mean_diff / v) / normal.cdf(mean_diff / v);
            
            mu[winner_idx] = mu[winner_idx] + sigma[winner_idx] * sigma[winner_idx] * c / v;
            mu[loser_idx] = mu[loser_idx] - sigma[loser_idx] * sigma[loser_idx] * c / v;
            
            let factor: f64 = 1.0 - sigma[winner_idx] * sigma[winner_idx] * sigma[loser_idx] * sigma[loser_idx] * c * (c + mean_diff / v) / (v * v);
            sigma[winner_idx] = sigma[winner_idx] * factor.sqrt();
            sigma[loser_idx] = sigma[loser_idx] * factor.sqrt();
            
            let winner_var: f64 = sigma[winner_idx] * sigma[winner_idx] + tau * tau;
            let loser_var: f64 = sigma[loser_idx] * sigma[loser_idx] + tau * tau;
            sigma[winner_idx] = winner_var.sqrt();
            sigma[loser_idx] = loser_var.sqrt();
        }
        
        for i in 0..n {
            let item = self.data.get_item_from_index(i)
                .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
            scores.insert(item, mu[i]);
        }
        
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_comparison_matrix_new() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let matrix = ComparisonMatrix::new(&items);
        
        assert_eq!(matrix.item_count(), 3);
        assert_eq!(matrix.total_comparisons(), 0);
    }
    
    #[test]
    fn test_add_comparison() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut matrix = ComparisonMatrix::new(&items);
        
        let comparison = Comparison {
            winner: "A".to_string(),
            loser: "B".to_string(),
        };
        
        matrix.add_comparison(&comparison).unwrap();
        
        assert_eq!(matrix.get_win_count(&"A".to_string(), &"B".to_string()).unwrap(), 1);
        assert_eq!(matrix.get_win_count(&"B".to_string(), &"A".to_string()).unwrap(), 0);
        assert_eq!(matrix.total_comparisons(), 1);
    }
    
    #[test]
    fn test_ranking_model_new() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let model = RankingModel::new(&items);
        
        assert_eq!(model.data.item_count(), 3);
        assert!(model.scores.is_none());
    }
    
    #[test]
    fn test_ranking_model_add_comparison() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut model = RankingModel::new(&items);
        
        let comparison = Comparison {
            winner: "A".to_string(),
            loser: "B".to_string(),
        };
        
        model.add_comparison(comparison).unwrap();
        
        assert_eq!(model.data.total_comparisons(), 1);
        assert!(model.scores.is_none());
    }
}
