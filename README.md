# Smart Hyperparameter Tuning Using Optimization

## Overview
This repository contains an advanced hyperparameter optimization framework that combines multiple sophisticated techniques: Genetic Algorithms, Bayesian Bandits, and Meta-Learning. The framework is demonstrated on a simplified AlexNet model for MNIST classification and achieves excellent accuracy results (98.86% on the final model).

## Key Features

### Hybrid Optimization Approach
The framework combines three powerful optimization techniques:
1. **Genetic Algorithm** - Evolutionary approach for broad parameter space exploration
2. **Bayesian Bandits** - Thompson sampling for efficient exploitation of promising regions
3. **Meta-Learning** - Neural network that learns to predict hyperparameter performance

### Results
The framework achieved excellent performance on MNIST classification:
- Final model accuracy: **98.86%**
- Best hyperparameters:
  - Learning rate: 0.0029
  - Batch size: 180
  - Dropout rate: 0.116
  - Weight decay: 0.0002

### Neural Network Architecture
- Simplified AlexNet adapted for MNIST dataset (28x28 grayscale images)
- Configurable dropout rates for regularization
- Early stopping based on gradient flatness

### Smart Training Features
- Gradient-based early stopping to prevent overfitting
- Optimized batch processing for faster hyperparameter evaluation
- Performance visualization for tracking optimization progress

## Technical Components

### `SimplifiedAlexNet` Class
A CNN based on AlexNet architecture but adapted for MNIST images:
- Convolutional layers with ReLU activation
- MaxPooling for downsampling
- Dropout for regularization
- Linear layers for classification

### `MetaLearner` Class
A neural network that learns to predict performance based on hyperparameter configurations:
- Takes hyperparameters as input
- Outputs predicted performance score
- Learns from actual training results

### `BayesianBandit` Class
Implementation of Thompson sampling for hyperparameter selection:
- Maintains historical performance of parameter combinations
- Samples from posterior distributions to select promising configurations
- Balances exploration and exploitation

### `GeneticOptimizer` Class
Evolutionary algorithm for hyperparameter optimization:
- Population-based search through parameter space
- Tournament selection for parent choice
- Single-point crossover for recombination
- Adaptive mutation based on parameter types
- Elitism to preserve best solutions

### `GradientEarlyStopping` Class
Early stopping mechanism based on loss gradient:
- Monitors loss changes over time
- Stops training when improvements plateau
- Prevents overfitting and reduces training time

## Optimization Progress

### Generation 2 Results
The genetic algorithm showed promising convergence in the second generation:
- Best Individual: Learning rate=0.0029, Batch size=254, Dropout rate=0.104, Weight decay=0.0008
- Test Accuracy: 98.58%

Example individual training progress:
```
Individual 4/10: (0.002403342844568322, 189, 0.11592158181031818, 0.000206849274179782)
Params: (0.002403342844568322, 189, 0.11592158181031818, 0.000206849274179782), Epoch: 1, Train Acc: 83.08%, Test Acc: 96.16%
Params: (0.002403342844568322, 189, 0.11592158181031818, 0.000206849274179782), Epoch: 2, Train Acc: 96.62%, Test Acc: 97.86%
Params: (0.002403342844568322, 189, 0.11592158181031818, 0.000206849274179782), Epoch: 3, Train Acc: 97.68%, Test Acc: 98.42%
Params: (0.002403342844568322, 189, 0.11592158181031818, 0.000206849274179782), Epoch: 4, Train Acc: 97.96%, Test Acc: 98.25%
Params: (0.002403342844568322, 189, 0.11592158181031818, 0.000206849274179782), Epoch: 5, Train Acc: 98.30%, Test Acc: 98.71%
```

### Generation 3 Results
The genetic algorithm further refined the parameters:
- Best Individual: Learning rate=0.0029, Batch size=180, Dropout rate=0.116, Weight decay=0.0002
- Test Accuracy: 98.44%

### Final Model Performance
After training the model with the best found hyperparameters for 10 epochs:
```
Params: (0.002854088011483561, 180, 0.11592158181031818, 0.000206849274179782), Epoch: 10, Train Acc: 98.69%, Test Acc: 98.86%
Final Model Performance - Loss: 0.0560, Accuracy: 98.86%
```

## Hyperparameters Optimized
The framework optimizes the following hyperparameters:
- Learning rate (range: 0.0001 to 0.01)
- Batch size (range: 32 to 256)
- Dropout rate (range: 0.1 to 0.7)
- Weight decay (range: 0.00001 to 0.001)

## Usage

```python
if __name__ == "__main__":
    print("Starting Hybrid Hyperparameter Tuning for AlexNet on MNIST")
    best_params, best_score, best_accuracy = smart_hyperparameter_tuning(max_iters=25, genetic_gens=3)
    
    print("\n=== Final Results ===")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Score: {-best_score:.4f}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    
    # Train a final model with the best parameters
    print("\nTraining final model with best parameters...")
    final_score, final_accuracy = train_model_with_params(best_params, max_epochs=10)
```

## Optimization Process
The framework follows a two-phase approach:

1. **Phase 1: Genetic Algorithm Exploration**
   - Creates an initial population of hyperparameter combinations
   - Evaluates each combination by training a model
   - Uses tournament selection, crossover, and mutation to evolve better hyperparameters
   - Tracks the best individual across generations

2. **Phase 2: Bayesian Bandit Refinement**
   - Takes the promising hyperparameter space identified by the genetic algorithm
   - Uses Thompson sampling to balance exploration and exploitation
   - Leverages the meta-learner to predict the performance of untested configurations
   - Efficiently searches the narrowed hyperparameter space

## Output and Visualization
The framework generates:
- Detailed logs of the optimization process
- A comprehensive visualization (`hybrid_tuning_results.png`) showing:
  - Overall accuracy and loss trends
  - Performance distributions across genetic generations
  - Bandit refinement progress

## Dependencies
- PyTorch
- NumPy
- SciPy
- Matplotlib
- torchvision

## Patent Pending
This advanced hyperparameter optimization approach is patent pending. The code is provided for research and educational purposes.

## License
[Your chosen license]

## Citation
If you use this code in your research, please cite:
```
[Your citation information]
```
