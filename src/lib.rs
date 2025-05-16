use statrs::distribution::ContinuousCDF;
use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::hash::Hash;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comparison<T: Display> {
    pub winner: T,
    pub loser: T,
}

impl<T: Display> fmt::Display for Comparison<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} > {}", self.winner, self.loser)
    }
}

#[derive(Error, Debug)]
pub enum AsapError<T: Display + Debug> {
    #[error("Item not found: {0}")]
    ItemNotFound(T),
    #[error("Item already exists: {0}")]
    ItemAlreadyExists(T),
    #[error("Invalid comparison: both items are the same")]
    InvalidComparison,
    #[error("Not enough comparisons to compute scores")]
    NotEnoughComparisons,
    #[error("Internal algorithm error: {0}")]
    InternalError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(
    serialize = "T: Clone + Debug + Eq + Hash + Send + Sync + 'static + serde::Serialize",
    deserialize = "T: Clone + Debug + Eq + Hash + Send + Sync + 'static + serde::de::DeserializeOwned"
)))]
pub struct ComparisonMatrix<T: Clone + Debug + Eq + Hash + Send + Sync + 'static> {
    item_indices: HashMap<T, usize>,
    index_to_item: Vec<T>,
    win_counts: Vec<Vec<usize>>,
    comparison_count: usize,
}

impl<T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static> ComparisonMatrix<T> {
    pub fn new(items: &[T]) -> Self {
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

    pub fn add_comparison(&mut self, comparison: &Comparison<T>) -> Result<(), AsapError<T>> {
        if comparison.winner == comparison.loser {
            return Err(AsapError::InvalidComparison);
        }

        let winner_idx = self
            .item_indices
            .get(&comparison.winner)
            .ok_or_else(|| AsapError::ItemNotFound(comparison.winner.clone()))?;
        let loser_idx = self
            .item_indices
            .get(&comparison.loser)
            .ok_or_else(|| AsapError::ItemNotFound(comparison.loser.clone()))?;

        self.win_counts[*winner_idx][*loser_idx] += 1;
        self.comparison_count += 1;

        Ok(())
    }

    pub fn get_win_count(&self, item_i: &T, item_j: &T) -> Result<usize, AsapError<T>> {
        let i_idx = self
            .item_indices
            .get(item_i)
            .ok_or_else(|| AsapError::ItemNotFound(item_i.clone()))?;
        let j_idx = self
            .item_indices
            .get(item_j)
            .ok_or_else(|| AsapError::ItemNotFound(item_j.clone()))?;

        Ok(self.win_counts[*i_idx][*j_idx])
    }

    pub fn get_comparison_count(
        &self,
        item_i: &T,
        item_j: &T,
    ) -> Result<usize, AsapError<T>> {
        let i_idx = self
            .item_indices
            .get(item_i)
            .ok_or_else(|| AsapError::ItemNotFound(item_i.clone()))?;
        let j_idx = self
            .item_indices
            .get(item_j)
            .ok_or_else(|| AsapError::ItemNotFound(item_j.clone()))?;

        Ok(self.win_counts[*i_idx][*j_idx] + self.win_counts[*j_idx][*i_idx])
    }

    pub fn item_count(&self) -> usize {
        self.index_to_item.len()
    }

    pub fn total_comparisons(&self) -> usize {
        self.comparison_count
    }

    pub fn items(&self) -> Vec<T> {
        self.index_to_item.clone()
    }

    pub fn get_item_index(&self, item: &T) -> Result<usize, AsapError<T>> {
        self.item_indices
            .get(item)
            .copied()
            .ok_or_else(|| AsapError::ItemNotFound(item.clone()))
    }

    pub fn get_item_from_index(&self, index: usize) -> Option<T> {
        self.index_to_item.get(index).cloned()
    }

    pub fn add_item(&mut self, item: T) -> Result<(), AsapError<T>> {
        if self.item_indices.contains_key(&item) {
            return Err(AsapError::ItemAlreadyExists(item));
        }

        let new_idx = self.index_to_item.len();

        self.item_indices.insert(item.clone(), new_idx);
        self.index_to_item.push(item);

        self.win_counts.push(vec![0; new_idx + 1]);

        for row in &mut self.win_counts[0..new_idx] {
            row.push(0);
        }

        Ok(())
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(
    serialize = "T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static + serde::Serialize",
    deserialize = "T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static + serde::de::DeserializeOwned"
)))]
pub struct RankingModel<T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static> {
    pub data: ComparisonMatrix<T>,
    pub scores: Option<HashMap<T, f64>>,
    approximate: bool,
    #[allow(dead_code)]
    selective_eig: bool,
}

impl<T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static> RankingModel<T> {
    pub fn new(items: &[T]) -> Self {
        RankingModel {
            data: ComparisonMatrix::new(items),
            scores: None,
            approximate: false,
            selective_eig: false,
        }
    }

    pub fn new_with_options(items: &[T], approximate: bool, selective_eig: bool) -> Self {
        RankingModel {
            data: ComparisonMatrix::new(items),
            scores: None,
            approximate,
            selective_eig,
        }
    }

    pub fn add_comparison(&mut self, comparison: Comparison<T>) -> Result<(), AsapError<T>> {
        self.data.add_comparison(&comparison)?;
        self.scores = None;
        Ok(())
    }

    pub fn add_item(&mut self, item: T) -> Result<(), AsapError<T>> {
        self.data.add_item(item)?;
        self.scores = None;
        Ok(())
    }

    pub fn get_ordering(&mut self) -> Result<Vec<T>, AsapError<T>> {
        let scores = self.get_scores()?;

        let mut items_with_scores: Vec<_> = scores.iter().collect();

        items_with_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(items_with_scores
            .into_iter()
            .map(|(item, _)| item.clone())
            .collect())
    }

    pub fn get_scores(&mut self) -> Result<HashMap<T, f64>, AsapError<T>> {
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

    pub fn suggest_comparisons(&self, max: usize) -> Result<Vec<(T, T)>, AsapError<T>> {
        if self.data.item_count() < 2 {
            return Err(AsapError::NotEnoughComparisons);
        }

        let mut pairs_with_gain = Vec::new();
        let items = self.data.items();
        let n = items.len();

        let current_scores = match self.scores {
            Some(ref s) => s.clone(),
            None => {
                // Calculate temporary scores if not available (e.g., simple win ratio)
                // This part might need a mutable self to call get_scores, or a non-mutating score estimation
                // For now, let's assume if scores are None, we compute them temporarily or use a default.
                // A proper solution might involve ensuring scores are computed or using a lightweight estimation.
                // The original code used a simple win ratio if scores were not present.
                let mut temp_scores = HashMap::new();
                for i_idx in 0..n {
                    let item_i = self.data.get_item_from_index(i_idx)
                        .ok_or_else(|| AsapError::InternalError("Invalid item index in suggest_comparisons".to_string()))?;
                    let mut wins = 0;
                    let mut total_comps = 0;
                    for j_idx in 0..n {
                        if i_idx == j_idx { continue; }
                        let item_j = self.data.get_item_from_index(j_idx)
                            .ok_or_else(|| AsapError::InternalError("Invalid item index in suggest_comparisons".to_string()))?;
                        wins += self.data.get_win_count(&item_i, &item_j)?;
                        total_comps += self.data.get_comparison_count(&item_i, &item_j)?;
                    }
                    temp_scores.insert(item_i.clone(), if total_comps > 0 { wins as f64 / total_comps as f64 } else { 0.5 });
                }
                temp_scores
            }
        };


        for i in 0..n {
            for j in (i + 1)..n {
                let item_i = &items[i];
                let item_j = &items[j];

                let score_i = *current_scores.get(item_i).unwrap_or(&0.5);
                let score_j = *current_scores.get(item_j).unwrap_or(&0.5);

                let prob_i_wins = 1.0 / (1.0 + (-10.0 * (score_i - score_j)).exp());

                let info_gain = -(prob_i_wins * prob_i_wins.ln()
                    + (1.0 - prob_i_wins) * (1.0 - prob_i_wins).ln());

                let comparison_count = self.data.get_comparison_count(item_i, item_j)?;
                let adjusted_gain = info_gain / (1.0 + 0.1 * comparison_count as f64);

                pairs_with_gain.push((adjusted_gain, (item_i.clone(), item_j.clone())));
            }
        }

        pairs_with_gain.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let result = pairs_with_gain
            .into_iter()
            .take(max)
            .map(|(_, pair)| pair)
            .collect();

        Ok(result)
    }

    pub fn ranking_confidence(&self) -> Result<f64, AsapError<T>> {
        let n = self.data.item_count();
        if n <= 1 {
            return Ok(1.0); // Only one item, so we're 100% confident
        }

        let current_scores = match self.scores {
            Some(ref s) => s.clone(),
             None => {
                // If scores are not computed, confidence might be based purely on comparison count
                let max_comparisons_possible = (n * (n - 1)) / 2;
                if max_comparisons_possible == 0 { return Ok(0.0); } // Avoid division by zero if n=0 or n=1 (already handled)
                let confidence_from_count = (self.data.total_comparisons() as f64) / (max_comparisons_possible as f64);
                // Sigmoid scaling for confidence
                return Ok(1.0 / (1.0 + (-5.0 * (confidence_from_count - 0.5)).exp()));
            }
        };

        let mut score_vec = Vec::with_capacity(n);
        let items = self.data.items();

        for item in &items {
            score_vec.push(*current_scores.get(item).unwrap_or(&0.5));
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
            return Ok(0.5); // Default confidence if no differences (e.g. all scores are same)
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

    pub fn is_sufficiently_confident(&self, threshold: f64) -> Result<bool, AsapError<T>> {
        let confidence = self.ranking_confidence()?;
        Ok(confidence >= threshold)
    }

    fn compute_accurate_scores(&self) -> Result<HashMap<T, f64>, AsapError<T>> {
        use nalgebra::DVector;
        use statrs::distribution::{Continuous, Normal};

        let n = self.data.item_count();
        let mut scores = HashMap::new();

        if n == 0 {
            return Ok(scores);
        }

        let mut mu = DVector::zeros(n);
        let mut sigma = DVector::from_element(n, 1.0);

        let beta = 0.1f64; // Skill variability
        let _tau = 0.05f64; // Dynamic factor (original paper, seems unused here in updates)

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let item_i = self
                    .data
                    .get_item_from_index(i)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
                let item_j = self
                    .data
                    .get_item_from_index(j)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;

                let wins_i_over_j = self.data.get_win_count(&item_i, &item_j)?;
                let wins_j_over_i = self.data.get_win_count(&item_j, &item_i)?;

                if wins_i_over_j == 0 && wins_j_over_i == 0 {
                    continue;
                }

                for _ in 0..wins_i_over_j {
                    let v = (2.0 * beta * beta + sigma[i] * sigma[i] + sigma[j] * sigma[j]).sqrt();
                    let mean_diff = mu[i] - mu[j];

                    let normal = Normal::new(0.0, 1.0).unwrap(); // Should handle potential error
                    let cdf_val = normal.cdf(mean_diff / v);
                    if cdf_val == 0.0 { continue; } // Avoid division by zero if cdf is 0
                    let c = v * normal.pdf(mean_diff / v) / cdf_val;


                    mu[i] += sigma[i] * sigma[i] * c / v;
                    mu[j] -= sigma[j] * sigma[j] * c / v;

                    let factor_val = sigma[i] * sigma[i] * sigma[j] * sigma[j] * c * (c + mean_diff / v) / (v * v);
                    let factor: f64 = (1.0 - factor_val).max(0.0); // Ensure factor is non-negative for sqrt
                    sigma[i] *= factor.sqrt();
                    sigma[j] *= factor.sqrt();
                }

                for _ in 0..wins_j_over_i {
                    let v = (2.0 * beta * beta + sigma[i] * sigma[i] + sigma[j] * sigma[j]).sqrt();
                    let mean_diff = mu[j] - mu[i];

                    let normal = Normal::new(0.0, 1.0).unwrap(); // Should handle potential error
                    let cdf_val = normal.cdf(mean_diff / v);
                    if cdf_val == 0.0 { continue; } // Avoid division by zero
                    let c = v * normal.pdf(mean_diff / v) / cdf_val;

                    mu[j] += sigma[j] * sigma[j] * c / v;
                    mu[i] -= sigma[i] * sigma[i] * c / v;

                    let factor_val = sigma[i] * sigma[i] * sigma[j] * sigma[j] * c * (c + mean_diff / v) / (v * v);
                    let factor: f64 = (1.0 - factor_val).max(0.0); // Ensure factor is non-negative
                    sigma[i] *= factor.sqrt();
                    sigma[j] *= factor.sqrt();
                }
            }
        }

        for i in 0..n {
            let item = self
                .data
                .get_item_from_index(i)
                .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
            scores.insert(item.clone(), mu[i]);
        }

        Ok(scores)
    }

    fn compute_approximate_scores(&self) -> Result<HashMap<T, f64>, AsapError<T>> {
        use nalgebra::DVector;
        use rand::prelude::*;
        use statrs::distribution::{Continuous, Normal};

        let n = self.data.item_count();
        let mut scores = HashMap::new();

        if n == 0 {
            return Ok(scores);
        }

        let mut mu = DVector::zeros(n);
        let mut sigma = DVector::from_element(n, 1.0);

        let beta = 0.1f64; // Skill variability
        let tau_sq = 0.05f64 * 0.05f64; // Dynamic factor squared

        let mut all_comparisons_indices = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let item_i_ref = self
                    .data
                    .get_item_from_index(i)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
                let item_j_ref = self
                    .data
                    .get_item_from_index(j)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;

                let wins_i_over_j = self.data.get_win_count(&item_i_ref, &item_j_ref)?;

                for _ in 0..wins_i_over_j {
                    all_comparisons_indices.push((i, j)); // Store indices
                }
            }
        }

        let mut rng = rand::thread_rng();
        all_comparisons_indices.shuffle(&mut rng);

        for (winner_idx, loser_idx) in all_comparisons_indices {
            let v = (2.0 * beta * beta
                + sigma[winner_idx] * sigma[winner_idx]
                + sigma[loser_idx] * sigma[loser_idx])
                .sqrt();

            let mean_diff = mu[winner_idx] - mu[loser_idx];

            let normal = Normal::new(0.0, 1.0).unwrap(); // Should handle error
            let cdf_val = normal.cdf(mean_diff / v);
            if cdf_val == 0.0 { continue; } // Avoid division by zero
            let c = v * normal.pdf(mean_diff / v) / cdf_val;

            mu[winner_idx] += sigma[winner_idx] * sigma[winner_idx] * c / v;
            mu[loser_idx] -= sigma[loser_idx] * sigma[loser_idx] * c / v;
            
            let factor_val = sigma[winner_idx] * sigma[winner_idx] * sigma[loser_idx] * sigma[loser_idx] * c * (c + mean_diff / v) / (v*v);
            let factor = (1.0 - factor_val).max(0.0); // Ensure non-negative

            sigma[winner_idx] = (sigma[winner_idx] * sigma[winner_idx] * factor + tau_sq).sqrt();
            sigma[loser_idx] = (sigma[loser_idx] * sigma[loser_idx] * factor + tau_sq).sqrt();
        }

        for i in 0..n {
            let item = self
                .data
                .get_item_from_index(i)
                .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
            scores.insert(item.clone(), mu[i]);
        }

        Ok(scores)
    }
}

#[cfg(feature = "serde")]
impl<T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static + serde::Serialize + serde::de::DeserializeOwned> RankingModel<T> {
    pub fn to_json(&self) -> Result<String, AsapError<T>> {
        serde_json::to_string(self)
            .map_err(|e| AsapError::SerializationError(format!("Failed to serialize: {}", e)))
    }

    pub fn from_json(json: &str) -> Result<Self, AsapError<T>> {
        serde_json::from_str(json)
            .map_err(|e| AsapError::SerializationError(format!("Failed to deserialize: {}", e)))
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), AsapError<T>> {
        let json = self.to_json()?;
        std::fs::write(path, json)
            .map_err(|e| AsapError::SerializationError(format!("Failed to write file: {}", e)))
    }

    pub fn load_from_file(path: &str) -> Result<Self, AsapError<T>> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| AsapError::SerializationError(format!("Failed to read file: {}", e)))?;
        Self::from_json(&json)
    }
}

#[cfg(not(feature = "serde"))]
impl<T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static> RankingModel<T> {
    // Define stubs or error-out for non-serde builds if these methods are called.
    // Or, conditionally compile these methods only when "serde" feature is enabled.
    // For now, these methods will only be available if "serde" is on, as per the cfg above.
    // If to_json etc. must exist always, they should return an error when serde is off.
    // Example stub (would require AsapError to be non-generic or handle this):
    // pub fn to_json(&self) -> Result<String, AsapError<T>> {
    //     Err(AsapError::SerializationError("Serde feature not enabled".to_string()))
    // }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_matrix_new() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let matrix = ComparisonMatrix::<String>::new(&items);

        assert_eq!(matrix.item_count(), 3);
        assert_eq!(matrix.total_comparisons(), 0);
    }

    #[test]
    fn test_add_comparison() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut matrix = ComparisonMatrix::<String>::new(&items);

        let comparison = Comparison {
            winner: "A".to_string(),
            loser: "B".to_string(),
        };

        matrix.add_comparison(&comparison).unwrap();

        assert_eq!(
            matrix
                .get_win_count(&"A".to_string(), &"B".to_string())
                .unwrap(),
            1
        );
        assert_eq!(
            matrix
                .get_win_count(&"B".to_string(), &"A".to_string())
                .unwrap(),
            0
        );
        assert_eq!(matrix.total_comparisons(), 1);
    }

    #[test]
    fn test_comparison_matrix_add_item() {
        let items = vec!["A".to_string(), "B".to_string()];
        let mut matrix = ComparisonMatrix::<String>::new(&items);

        matrix.add_item("C".to_string()).unwrap();

        assert_eq!(matrix.item_count(), 3);
        assert_eq!(
            matrix
                .get_win_count(&"A".to_string(), &"C".to_string())
                .unwrap(),
            0
        );
        assert_eq!(
            matrix
                .get_win_count(&"C".to_string(), &"A".to_string())
                .unwrap(),
            0
        );

        let result = matrix.add_item("A".to_string());
        assert!(result.is_err());

        let comparison = Comparison {
            winner: "C".to_string(),
            loser: "A".to_string(),
        };
        matrix.add_comparison(&comparison).unwrap();

        assert_eq!(
            matrix
                .get_win_count(&"C".to_string(), &"A".to_string())
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_ranking_model_new() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let model = RankingModel::<String>::new(&items);

        assert_eq!(model.data.item_count(), 3);
        assert!(model.scores.is_none());
    }

    #[test]
    fn test_ranking_model_add_comparison() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut model = RankingModel::<String>::new(&items);

        let comparison = Comparison {
            winner: "A".to_string(),
            loser: "B".to_string(),
        };

        model.add_comparison(comparison).unwrap();

        assert_eq!(model.data.total_comparisons(), 1);
        assert!(model.scores.is_none());
    }

    #[test]
    fn test_ranking_model_add_item() {
        let items = vec!["A".to_string(), "B".to_string()];
        let mut model = RankingModel::<String>::new(&items);

        model
            .add_comparison(Comparison {
                winner: "A".to_string(),
                loser: "B".to_string(),
            })
            .unwrap();
        model
            .add_comparison(Comparison {
                winner: "A".to_string(),
                loser: "B".to_string(),
            })
            .unwrap();

        let scores_before = model.get_scores().unwrap();
        assert!(scores_before.contains_key(&"A".to_string()));
        assert!(scores_before.contains_key(&"B".to_string()));

        model.add_item("C".to_string()).unwrap();

        assert!(model.scores.is_none());

        model
            .add_comparison(Comparison {
                winner: "C".to_string(),
                loser: "A".to_string(),
            })
            .unwrap();

        let scores_after = model.get_scores().unwrap();
        assert!(scores_after.contains_key(&"A".to_string()));
        assert!(scores_after.contains_key(&"B".to_string()));
        assert!(scores_after.contains_key(&"C".to_string()));

        assert!(
            scores_after.get(&"C".to_string()).unwrap()
                > scores_after.get(&"A".to_string()).unwrap()
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serialization_deserialization() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut model = RankingModel::<String>::new(&items);

        model
            .add_comparison(Comparison {
                winner: "A".to_string(),
                loser: "B".to_string(),
            })
            .unwrap();
        model
            .add_comparison(Comparison {
                winner: "B".to_string(),
                loser: "C".to_string(),
            })
            .unwrap();
        model
            .add_comparison(Comparison {
                winner: "A".to_string(),
                loser: "C".to_string(),
            })
            .unwrap();

        let original_scores = model.get_scores().unwrap();

        let json = model.to_json().unwrap();

        let mut deserialized_model = RankingModel::<String>::from_json(&json).unwrap();

        assert_eq!(
            deserialized_model.data.item_count(),
            model.data.item_count()
        );
        assert_eq!(
            deserialized_model.data.total_comparisons(),
            model.data.total_comparisons()
        );

        let deserialized_scores = deserialized_model.get_scores().unwrap();
        for (item, score) in &original_scores {
            assert!(deserialized_scores.contains_key(item));
            assert!((deserialized_scores.get(item).unwrap() - score).abs() < 1e-6);
        }
    }
}
