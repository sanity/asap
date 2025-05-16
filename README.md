# ASAP: Active Sampling for Pairwise Comparisons

A Rust implementation of the ASAP algorithm for active sampling in pairwise comparison preference aggregation. The algorithm offers high accuracy for inferring scores while minimizing the number of comparisons needed.

Based on the paper: "Active Sampling for Pairwise Comparisons via Approximate Message Passing and Information Gain Maximization" by A. Mikhailiuk, C. Wilmot, M. Perez-Ortiz, D. Yue and R. K. Mantiuk (2020).

## Features

- Accurate score inference from pairwise comparisons
- Suggestion of most informative comparisons to perform next
- Confidence estimation for the current ranking
- Support for both accurate and approximate algorithm versions

## Usage

```rust
use asap::{RankingModel, Comparison};

// Create a new model with items
let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
let mut model = RankingModel::<String>::new(&items);

// Add pairwise comparisons
model.add_comparison(Comparison::<String> {
    winner: "A".to_string(),
    loser: "B".to_string(),
}).unwrap();

model.add_comparison(Comparison::<String> {
    winner: "B".to_string(),
    loser: "C".to_string(),
}).unwrap();

// Get inferred scores
let scores = model.get_scores().unwrap();
println!("Scores: {:?}", scores);

// Get ordered ranking
let ranking = model.get_ordering().unwrap();
println!("Ranking: {:?}", ranking);

// Get suggestions for next comparisons
let suggestions = model.suggest_comparisons(3).unwrap();
println!("Suggested comparisons: {:?}", suggestions);

// Check if we have enough confidence in the ranking
let is_confident = model.is_sufficiently_confident(0.8).unwrap();
println!("Is ranking confident: {}", is_confident);
```

## Implementation Details

This implementation provides both the accurate and approximate versions of the ASAP algorithm:

1. **Accurate Version**: Uses full posterior updates for maximum accuracy
2. **Approximate Version**: Uses online posterior updates for reduced computation cost

The accurate version is used by default, but the approximate version can be enabled for larger datasets where performance is a concern.

## Examples

The repository includes an example binary that demonstrates:
- Basic usage of the ASAP algorithm
- Score recovery from pairwise comparisons with noise

Run the example with:
```
cargo run --bin example
```

## Testing

The implementation includes both unit tests and integration tests:
- Unit tests verify the correctness of individual components
- Integration tests validate the algorithm's ability to recover scores from pairwise comparisons

Run the tests with:
```
cargo test
```

## License

MIT
