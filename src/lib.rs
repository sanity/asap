use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::hash::Hash;
use thiserror::Error;

/// Represents a pairwise comparison between two items
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

/// Error types for ASAP operations
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

/// Matrix storing pairwise comparison results between items
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: Clone + Debug + Eq + Hash + Send + Sync + 'static + serde::Serialize",
        deserialize = "T: Clone + Debug + Eq + Hash + Send + Sync + 'static + serde::de::DeserializeOwned"
    ))
)]
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

    pub fn get_comparison_count(&self, item_i: &T, item_j: &T) -> Result<usize, AsapError<T>> {
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

    pub fn remove_item(&mut self, item_to_remove: &T) -> Result<(), AsapError<T>> {
        let removed_idx = match self.item_indices.get(item_to_remove) {
            Some(&idx) => idx,
            None => return Err(AsapError::ItemNotFound(item_to_remove.clone())),
        };

        let n_before_removal = self.index_to_item.len();
        let mut comps_to_remove = 0;

        // Calculate comparisons involving the removed item
        for j in 0..n_before_removal {
            if j == removed_idx {
                // Sum wins by the removed item over others
                for k in 0..n_before_removal {
                    if k != removed_idx {
                        comps_to_remove += self.win_counts[removed_idx][k];
                    }
                }
            } else {
                // Sum wins by other items over the removed item
                comps_to_remove += self.win_counts[j][removed_idx];
            }
        }

        // Remove the item from index_to_item
        self.index_to_item.remove(removed_idx);

        // Remove the corresponding row and column from win_counts
        self.win_counts.remove(removed_idx);
        for row in &mut self.win_counts {
            row.remove(removed_idx);
        }

        // Rebuild item_indices
        self.item_indices.clear();
        for (idx, item) in self.index_to_item.iter().enumerate() {
            self.item_indices.insert(item.clone(), idx);
        }

        self.comparison_count -= comps_to_remove;

        Ok(())
    }
}

/// Main ASAP ranking model for inferring scores from pairwise comparisons
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static + serde::Serialize",
        deserialize = "T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static + serde::de::DeserializeOwned"
    ))
)]
pub struct RankingModel<T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static> {
    pub data: ComparisonMatrix<T>,
    pub scores: Option<HashMap<T, f64>>,
    approximate: bool,
    #[allow(dead_code)]
    selective_eig: bool,
}

impl<T: Clone + Debug + Eq + Hash + Display + Send + Sync + 'static> RankingModel<T> {
    /// Create a new RankingModel with the given items
    pub fn new(items: &[T]) -> Self {
        RankingModel {
            data: ComparisonMatrix::new(items),
            scores: None,
            approximate: false,
            selective_eig: false,
        }
    }

    /// Create a new RankingModel with custom options
    pub fn new_with_options(items: &[T], approximate: bool, selective_eig: bool) -> Self {
        RankingModel {
            data: ComparisonMatrix::new(items),
            scores: None,
            approximate,
            selective_eig,
        }
    }

    /// Add a pairwise comparison result
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

    pub fn remove_item(&mut self, item: &T) -> Result<(), AsapError<T>> {
        self.data.remove_item(item)?;
        self.scores = None; // Scores become invalid
        Ok(())
    }

    /// Get the items ordered by their inferred scores (highest to lowest)
    pub fn get_ordering(&mut self) -> Result<Vec<T>, AsapError<T>> {
        let scores = self.get_scores()?;

        let mut items_with_scores: Vec<_> = scores.iter().collect();

        items_with_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(items_with_scores
            .into_iter()
            .map(|(item, _)| item.clone())
            .collect())
    }

    /// Get the inferred scores for all items
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

    /// Suggest the most informative comparisons to perform next
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
                    let item_i = self.data.get_item_from_index(i_idx).ok_or_else(|| {
                        AsapError::InternalError(
                            "Invalid item index in suggest_comparisons".to_string(),
                        )
                    })?;
                    let mut wins = 0;
                    let mut total_comps = 0;
                    for j_idx in 0..n {
                        if i_idx == j_idx {
                            continue;
                        }
                        let item_j = self.data.get_item_from_index(j_idx).ok_or_else(|| {
                            AsapError::InternalError(
                                "Invalid item index in suggest_comparisons".to_string(),
                            )
                        })?;
                        wins += self.data.get_win_count(&item_i, &item_j)?;
                        total_comps += self.data.get_comparison_count(&item_i, &item_j)?;
                    }
                    temp_scores.insert(
                        item_i.clone(),
                        if total_comps > 0 {
                            wins as f64 / total_comps as f64
                        } else {
                            0.5
                        },
                    );
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

    /// Confidence metric calibrated for Bradley-Terry MLE.
    ///
    /// Measures two things:
    /// 1. **Coverage** (50% weight): fraction of items with >= MIN_COMPARISONS_PER_ITEM
    ///    comparisons. BT needs ~5-10 per item, not N² total.
    /// 2. **Discrimination** (50% weight): whether scores are spread out enough to
    ///    distinguish items. Uses coefficient of variation of the BT strengths.
    pub fn ranking_confidence(&self) -> Result<f64, AsapError<T>> {
        let n = self.data.item_count();
        if n <= 1 {
            return Ok(1.0);
        }

        const MIN_COMPARISONS_PER_ITEM: usize = 5;

        // Coverage: fraction of items with enough comparisons
        let items = self.data.items();
        let mut well_compared = 0usize;
        for i in 0..n {
            let item_i = self.data.get_item_from_index(i)
                .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
            let mut total = 0usize;
            for j in 0..n {
                if i == j { continue; }
                let item_j = self.data.get_item_from_index(j)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
                total += self.data.get_comparison_count(&item_i, &item_j)?;
            }
            if total >= MIN_COMPARISONS_PER_ITEM {
                well_compared += 1;
            }
        }
        let coverage = well_compared as f64 / n as f64;

        // Discrimination: do scores actually differentiate items?
        let current_scores = match self.scores {
            Some(ref s) => s.clone(),
            None => {
                // No scores computed yet — confidence based on coverage alone
                return Ok(coverage * 0.5);
            }
        };

        let score_vec: Vec<f64> = items.iter()
            .map(|item| *current_scores.get(item).unwrap_or(&0.0))
            .collect();

        let mean = score_vec.iter().sum::<f64>() / n as f64;
        let variance = score_vec.iter()
            .map(|s| (s - mean) * (s - mean))
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // CV > 0.5 means good discrimination; saturates via sigmoid
        let cv = if mean.abs() > 1e-10 { std_dev / mean.abs() } else { std_dev };
        let discrimination = 1.0 / (1.0 + (-5.0 * (cv - 0.3)).exp());

        let confidence = 0.5 * coverage + 0.5 * discrimination;
        Ok(confidence)
    }

    /// Check if the current ranking has sufficient confidence (threshold: 0.0 to 1.0)
    pub fn is_sufficiently_confident(&self, threshold: f64) -> Result<bool, AsapError<T>> {
        let confidence = self.ranking_confidence()?;
        Ok(confidence >= threshold)
    }

    /// Bradley-Terry MLE via the MM (minorization-maximization) algorithm.
    /// Iterates until convergence, producing scores that are independent of
    /// item ordering. P(i beats j) = score_i / (score_i + score_j).
    /// Final scores are converted to log-scale for easier interpretation.
    fn compute_accurate_scores(&self) -> Result<HashMap<T, f64>, AsapError<T>> {
        let n = self.data.item_count();
        let mut scores = HashMap::new();

        if n == 0 {
            return Ok(scores);
        }

        // Build win matrix and total-games-against matrix
        let mut wins = vec![vec![0usize; n]; n];
        let mut total_wins = vec![0usize; n]; // total wins for item i
        let mut games_against = vec![vec![0usize; n]; n]; // n_ij = games between i and j

        for i in 0..n {
            let item_i = self
                .data
                .get_item_from_index(i)
                .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
            for j in (i + 1)..n {
                let item_j = self
                    .data
                    .get_item_from_index(j)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;

                let w_ij = self.data.get_win_count(&item_i, &item_j)?;
                let w_ji = self.data.get_win_count(&item_j, &item_i)?;

                wins[i][j] = w_ij;
                wins[j][i] = w_ji;
                total_wins[i] += w_ij;
                total_wins[j] += w_ji;
                let n_ij = w_ij + w_ji;
                games_against[i][j] = n_ij;
                games_against[j][i] = n_ij;
            }
        }

        // Initialize all strengths to 1.0
        let mut p = vec![1.0f64; n];

        const MAX_ITER: usize = 1000;
        const TOL: f64 = 1e-8;

        for _iter in 0..MAX_ITER {
            let mut p_new = vec![0.0f64; n];
            let mut max_change = 0.0f64;

            for i in 0..n {
                if total_wins[i] == 0 {
                    // Item never won — assign a small strength to avoid zero
                    p_new[i] = TOL;
                    continue;
                }

                // MM update: p_i = w_i / sum_j(n_ij / (p_i + p_j))
                let mut denom = 0.0f64;
                for j in 0..n {
                    if i == j || games_against[i][j] == 0 {
                        continue;
                    }
                    denom += games_against[i][j] as f64 / (p[i] + p[j]);
                }

                if denom > 0.0 {
                    p_new[i] = total_wins[i] as f64 / denom;
                } else {
                    p_new[i] = p[i];
                }
            }

            // Normalize so geometric mean = 1 (prevents drift)
            let log_sum: f64 = p_new.iter().map(|x| x.max(TOL).ln()).sum();
            let log_mean = log_sum / n as f64;
            let scale = (-log_mean).exp();
            for x in &mut p_new {
                *x *= scale;
            }

            // Check convergence
            for i in 0..n {
                let change = ((p_new[i] - p[i]) / p[i].max(TOL)).abs();
                if change > max_change {
                    max_change = change;
                }
            }

            p = p_new;

            if max_change < TOL {
                break;
            }
        }

        // Convert to log-scale scores for consistency with the rest of the system
        for (i, &pi) in p.iter().enumerate().take(n) {
            let item = self
                .data
                .get_item_from_index(i)
                .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
            scores.insert(item.clone(), pi.max(TOL).ln());
        }

        Ok(scores)
    }

    /// Approximate scoring: same Bradley-Terry MM algorithm but with fewer
    /// iterations for speed. Still order-independent unlike the old TrueSkill approach.
    fn compute_approximate_scores(&self) -> Result<HashMap<T, f64>, AsapError<T>> {
        let n = self.data.item_count();
        let mut scores = HashMap::new();

        if n == 0 {
            return Ok(scores);
        }

        // Build win/games matrices (same as accurate)
        let mut total_wins = vec![0usize; n];
        let mut games_against = vec![vec![0usize; n]; n];

        for i in 0..n {
            let item_i = self
                .data
                .get_item_from_index(i)
                .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
            for j in (i + 1)..n {
                let item_j = self
                    .data
                    .get_item_from_index(j)
                    .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;

                let w_ij = self.data.get_win_count(&item_i, &item_j)?;
                let w_ji = self.data.get_win_count(&item_j, &item_i)?;

                total_wins[i] += w_ij;
                total_wins[j] += w_ji;
                let n_ij = w_ij + w_ji;
                games_against[i][j] = n_ij;
                games_against[j][i] = n_ij;
            }
        }

        let mut p = vec![1.0f64; n];
        const MAX_ITER: usize = 100; // Fewer iterations than accurate
        const TOL: f64 = 1e-6;

        for _iter in 0..MAX_ITER {
            let mut p_new = vec![0.0f64; n];
            let mut max_change = 0.0f64;

            for i in 0..n {
                if total_wins[i] == 0 {
                    p_new[i] = TOL;
                    continue;
                }

                let mut denom = 0.0f64;
                for j in 0..n {
                    if i == j || games_against[i][j] == 0 {
                        continue;
                    }
                    denom += games_against[i][j] as f64 / (p[i] + p[j]);
                }

                if denom > 0.0 {
                    p_new[i] = total_wins[i] as f64 / denom;
                } else {
                    p_new[i] = p[i];
                }
            }

            let log_sum: f64 = p_new.iter().map(|x| x.max(TOL).ln()).sum();
            let log_mean = log_sum / n as f64;
            let scale = (-log_mean).exp();
            for x in &mut p_new {
                *x *= scale;
            }

            for i in 0..n {
                let change = ((p_new[i] - p[i]) / p[i].max(TOL)).abs();
                if change > max_change {
                    max_change = change;
                }
            }

            p = p_new;

            if max_change < TOL {
                break;
            }
        }

        for (i, &pi) in p.iter().enumerate().take(n) {
            let item = self
                .data
                .get_item_from_index(i)
                .ok_or_else(|| AsapError::InternalError("Invalid item index".to_string()))?;
            scores.insert(item.clone(), pi.max(TOL).ln());
        }

        Ok(scores)
    }
}

#[cfg(feature = "serde")]
impl<
    T: Clone
        + Debug
        + Eq
        + Hash
        + Display
        + Send
        + Sync
        + 'static
        + serde::Serialize
        + serde::de::DeserializeOwned,
> RankingModel<T>
{
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

    #[test]
    fn test_comparison_matrix_remove_item() {
        let items_initial = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut matrix = ComparisonMatrix::<String>::new(&items_initial);

        matrix
            .add_comparison(&Comparison {
                winner: "A".to_string(),
                loser: "B".to_string(),
            })
            .unwrap(); // A > B (1)
        matrix
            .add_comparison(&Comparison {
                winner: "B".to_string(),
                loser: "C".to_string(),
            })
            .unwrap(); // B > C (2)
        matrix
            .add_comparison(&Comparison {
                winner: "A".to_string(),
                loser: "C".to_string(),
            })
            .unwrap(); // A > C (3)

        assert_eq!(matrix.item_count(), 3);
        assert_eq!(matrix.total_comparisons(), 3);

        // Remove "B"
        matrix.remove_item(&"B".to_string()).unwrap();

        assert_eq!(matrix.item_count(), 2);
        // Comparisons involving B (A>B, B>C) are removed. A>C remains.
        // A>B means win_counts[A][B] = 1. B>C means win_counts[B][C] = 1.
        // comps_to_remove for B:
        // wins by B: win_counts[B_old_idx][C_old_idx] = 1
        // wins over B: win_counts[A_old_idx][B_old_idx] = 1
        // Total comps_to_remove = 1 (A>B) + 1 (B>C) = 2.
        // So, 3 - 2 = 1 comparison should remain.
        assert_eq!(matrix.total_comparisons(), 1);

        assert!(matrix.get_item_index(&"A".to_string()).is_ok());
        assert!(matrix.get_item_index(&"C".to_string()).is_ok());
        assert!(matrix.get_item_index(&"B".to_string()).is_err());

        // Check remaining win count A > C
        assert_eq!(
            matrix
                .get_win_count(&"A".to_string(), &"C".to_string())
                .unwrap(),
            1
        );

        // Check indices are updated
        let a_idx = matrix.get_item_index(&"A".to_string()).unwrap();
        let c_idx = matrix.get_item_index(&"C".to_string()).unwrap();
        assert!((a_idx == 0 && c_idx == 1) || (a_idx == 1 && c_idx == 0));

        // Try removing a non-existent item
        let result = matrix.remove_item(&"D".to_string());
        assert!(matches!(result, Err(AsapError::ItemNotFound(_))));

        // Remove "A"
        matrix.remove_item(&"A".to_string()).unwrap();
        assert_eq!(matrix.item_count(), 1);
        assert_eq!(matrix.total_comparisons(), 0);
        assert!(matrix.get_item_index(&"C".to_string()).is_ok());

        // Remove "C" (last item)
        matrix.remove_item(&"C".to_string()).unwrap();
        assert_eq!(matrix.item_count(), 0);
        assert_eq!(matrix.total_comparisons(), 0);
    }

    #[test]
    fn test_ranking_model_remove_item() {
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

        // Populate scores
        let scores_before_removal = model.get_scores().unwrap();
        assert!(scores_before_removal.contains_key(&"A".to_string()));
        assert!(scores_before_removal.contains_key(&"B".to_string()));
        assert!(scores_before_removal.contains_key(&"C".to_string()));
        assert!(model.scores.is_some());

        // Remove "B"
        model.remove_item(&"B".to_string()).unwrap();

        assert_eq!(model.data.item_count(), 2);
        assert!(model.scores.is_none()); // Scores should be cleared

        // Get scores again
        let scores_after_removal = model.get_scores().unwrap();
        assert!(scores_after_removal.contains_key(&"A".to_string()));
        assert!(scores_after_removal.contains_key(&"C".to_string()));
        assert!(!scores_after_removal.contains_key(&"B".to_string()));

        // Check ordering
        let ordering = model.get_ordering().unwrap();
        assert_eq!(ordering.len(), 2);
        assert!(ordering.contains(&"A".to_string()));
        assert!(ordering.contains(&"C".to_string()));

        // Try removing a non-existent item
        let result = model.remove_item(&"D".to_string());
        assert!(matches!(result, Err(AsapError::ItemNotFound(_))));
    }
}
