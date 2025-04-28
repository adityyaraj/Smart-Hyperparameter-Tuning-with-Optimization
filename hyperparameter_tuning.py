import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import copy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define a simplified AlexNet for MNIST
class SimplifiedAlexNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimplifiedAlexNet, self).__init__()
        # MNIST is 1x28x28, so we adapt AlexNet for smaller input
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Meta-Learner to remember past hyperparameter performance
class MetaLearner(nn.Module):
    def __init__(self, input_size):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Define Bayesian Multi-Armed Bandit for Hyperparameter Selection
# The fix is in the BayesianBandit class
class BayesianBandit:
    def __init__(self, param_space):
        self.param_space = param_space
        # Initialize history with string representation of tuples as keys
        self.history = {}
        for p in param_space:
            self.history[str(p)] = []
    
    def sample(self, method="thompson"):
        """Select hyperparameter using Bayesian Exploration"""
        if method == "thompson":
            sampled_means = []
            for p in self.param_space:
                # Use str(p) as the key for history
                rewards = self.history.get(str(p), [])
                if rewards:
                    mean, std = np.mean(rewards), np.std(rewards) + 1e-4
                else:
                    mean, std = 0, 1
                sampled_means.append(norm.rvs(mean, std))  # Thompson Sampling
            return self.param_space[np.argmax(sampled_means)]
        else:
            return random.choice(self.param_space)
    
    def update(self, params, score):
        """Update the performance history"""
        # Create the key if it doesn't exist
        key = str(params)
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(score)
# Genetic Algorithm for Hyperparameter Optimization
class GeneticOptimizer:
    def __init__(self, param_bounds, population_size=10, mutation_rate=0.2, crossover_rate=0.7):
        """
        param_bounds: Dictionary with parameter names as keys and (min, max, type) as values
                     type can be 'float', 'int', or 'categorical' (for list of options)
        """
        self.param_bounds = param_bounds
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.generation = 0
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Create initial random population"""
        self.population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val, param_type) in self.param_bounds.items():
                if param_type == 'float':
                    individual[param] = min_val + random.random() * (max_val - min_val)
                elif param_type == 'int':
                    individual[param] = random.randint(min_val, max_val)
                elif param_type == 'categorical':
                    individual[param] = random.choice(min_val)  # min_val is a list of options
            self.population.append(individual)
    
    def _selection(self, fitness_scores):
        """Tournament selection"""
        selected = []
        for _ in range(self.population_size):
            # Select 3 random individuals for tournament
            indices = random.sample(range(self.population_size), 3)
            tournament = [(i, fitness_scores[i]) for i in indices]
            winner_idx = max(tournament, key=lambda x: x[1])[0]
            selected.append(copy.deepcopy(self.population[winner_idx]))
        return selected
    
    def _crossover(self, parent1, parent2):
        """Single point crossover between two parents"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        child = {}
        # Determine crossover point
        params = list(parent1.keys())
        crossover_point = random.randint(0, len(params) - 1)
        
        for i, param in enumerate(params):
            if i <= crossover_point:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        
        return child
    
    def _mutate(self, individual):
        """Mutate individual with probability mutation_rate"""
        mutated = copy.deepcopy(individual)
        
        for param, (min_val, max_val, param_type) in self.param_bounds.items():
            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                if param_type == 'float':
                    # Gaussian mutation for float
                    current = mutated[param]
                    range_width = max_val - min_val
                    mutation = random.gauss(0, range_width * 0.1)
                    mutated[param] = max(min_val, min(max_val, current + mutation))
                elif param_type == 'int':
                    # Add or subtract by 1 (or more) for integers
                    current = mutated[param]
                    mutation = random.choice([-1, 1]) * random.randint(1, max(1, int((max_val - min_val) * 0.1)))
                    mutated[param] = max(min_val, min(max_val, current + mutation))
                elif param_type == 'categorical':
                    # Random choice for categorical
                    mutated[param] = random.choice(min_val)
        
        return mutated
    
    def evolve(self, fitness_scores):
        """Evolve the population based on fitness scores"""
        # Update best individual
        max_idx = np.argmax(fitness_scores)
        if fitness_scores[max_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[max_idx]
            self.best_individual = copy.deepcopy(self.population[max_idx])
        
        # Selection
        selected = self._selection(fitness_scores)
        
        # Crossover and Mutation
        new_population = []
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size:
                parent1, parent2 = selected[i], selected[i + 1]
                child1 = self._crossover(parent1, parent2)
                child2 = self._crossover(parent2, parent1)
                
                new_population.append(self._mutate(child1))
                new_population.append(self._mutate(child2))
            else:
                # Handle odd population size
                new_population.append(self._mutate(selected[i]))
        
        # Elitism: Replace worst individual with best from previous generation
        if self.best_individual:
            worst_idx = np.argmin(fitness_scores)
            new_population[worst_idx] = copy.deepcopy(self.best_individual)
        
        self.population = new_population
        self.generation += 1
        
        # Save fitness history
        self.fitness_history.append(np.mean(fitness_scores))
        
        return self.population, self.best_individual, self.best_fitness
    
    def get_population_as_tuples(self, param_keys):
        """Convert population dictionary to tuples in specific order"""
        return [tuple(ind[key] for key in param_keys) for ind in self.population]

# Early Stopping Based on Gradient Flatness
class GradientEarlyStopping:
    def __init__(self, patience=5, min_change=1e-3):
        self.patience = patience
        self.min_change = min_change
        self.history = []
    
    def check_stop(self, loss):
        """Check if training should be stopped early"""
        self.history.append(loss)
        if len(self.history) > self.patience:
            grad_changes = np.abs(np.diff(self.history[-self.patience:]))
            if np.mean(grad_changes) < self.min_change:
                return True
        return False

# Helper function to load MNIST data
def load_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Train function for a single epoch
def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Early batch limit for faster iterations during hyperparameter search
        if batch_idx >= 50:  # Process only first 50 batches for quick evaluation
            break
    
    return running_loss / (batch_idx + 1), 100. * correct / total

# Test function for evaluation
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

# Training function with hyperparameters
def train_model_with_params(params, max_epochs=5):
    # Extract parameters
    learning_rate, batch_size, dropout_rate, weight_decay = params
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, test_loader = load_mnist_data(int(batch_size))
    
    # Initialize model
    model = SimplifiedAlexNet(dropout_rate=dropout_rate).to(device)
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopper = GradientEarlyStopping(patience=3, min_change=1e-3)
    
    # Train
    losses = []
    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        losses.append(test_loss)
        print(f"Params: {params}, Epoch: {epoch+1}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if early_stopper.check_stop(test_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Return negative loss as the score (higher is better)
    return -np.mean(losses), test_acc

# Main Smart Tuning Process with Hybrid Approach
def smart_hyperparameter_tuning(max_iters=30, genetic_gens=5):
    """
    Performs Smart Hyperparameter Tuning combining Genetic Algorithm, 
    Bayesian Bandits and Meta-learning for AlexNet on MNIST
    """
    print("Starting hybrid hyperparameter tuning with Genetic Algorithms, Bayesian Bandits, and Meta-learning")
    
    # Define parameter bounds for genetic algorithm
    param_bounds = {
        'learning_rate': (0.0001, 0.01, 'float'),
        'batch_size': (32, 256, 'int'),
        'dropout_rate': (0.1, 0.7, 'float'),
        'weight_decay': (0.00001, 0.001, 'float')
    }
    
    # Initialize genetic optimizer
    genetic_opt = GeneticOptimizer(
        param_bounds=param_bounds,
        population_size=10,
        mutation_rate=0.2,
        crossover_rate=0.7
    )
    
    # Initialize meta-learner
    meta_learner = MetaLearner(input_size=4)  # 4 hyperparameters
    meta_optimizer = optim.Adam(meta_learner.parameters(), lr=0.01)
    
    # Track results
    all_results = []
    best_params, best_score, best_accuracy = None, float('-inf'), 0
    
    # Phase 1: Genetic Algorithm Exploration
    print("\n=== Phase 1: Genetic Algorithm Exploration ===")
    for gen in range(genetic_gens):
        print(f"\n--- Generation {gen+1}/{genetic_gens} ---")
        
        # Convert population to parameter tuples
        param_keys = ['learning_rate', 'batch_size', 'dropout_rate', 'weight_decay']
        population_tuples = genetic_opt.get_population_as_tuples(param_keys)
        
        # Evaluate population
        fitness_scores = []
        accuracies = []
        
        for i, params in enumerate(population_tuples):
            print(f"Individual {i+1}/{len(population_tuples)}: {params}")
            start_time = time.time()
            score, accuracy = train_model_with_params(params)
            end_time = time.time()
            fitness_scores.append(score)
            accuracies.append(accuracy)
            
            # Update best parameters
            if score > best_score:
                best_params, best_score, best_accuracy = params, score, accuracy
            
            # Track results
            all_results.append({
                'phase': 'genetic',
                'generation': gen + 1,
                'individual': i + 1,
                'params': params,
                'score': -score,  # Convert back to loss
                'accuracy': accuracy,
                'time': end_time - start_time
            })
            
            # Update meta-learner
            param_tensor = torch.tensor(params, dtype=torch.float32)
            target_score = torch.tensor(score, dtype=torch.float32)
            
            meta_optimizer.zero_grad()
            predicted_score = meta_learner(param_tensor)
            meta_loss = F.mse_loss(predicted_score, target_score.view(-1, 1))
            meta_loss.backward()
            meta_optimizer.step()
        
        # Evolve population
        genetic_opt.evolve(fitness_scores)
        
        print(f"Generation {gen+1} best: {genetic_opt.best_individual}")
        print(f"Generation {gen+1} best fitness: {genetic_opt.best_fitness:.4f}")
    
    # Convert final genetic population to parameter space for Bayesian Bandit
    final_population = genetic_opt.get_population_as_tuples(param_keys)
    
    # Add best individuals from each generation to the param space
    param_space = list(set(final_population))
    
    # Phase 2: Bayesian Bandit Refinement
    print("\n=== Phase 2: Bayesian Bandit Refinement ===")
    bandit = BayesianBandit(param_space)
    
    # Initialize bandit with genetic results
    for result in all_results:
        if result['phase'] == 'genetic':
            bandit.update(result['params'], -result['score'])  # Convert loss back to score
    
    remaining_iters = max_iters - genetic_gens * len(final_population)
    for i in range(remaining_iters):
        print(f"\n--- Iteration {i+1}/{remaining_iters} ---")
        
        # Sample parameters using Bayesian Bandit
        params = bandit.sample()
        print(f"Selected params: {params}")
        
        # Use meta-learner to predict performance
        param_tensor = torch.tensor(params, dtype=torch.float32)
        with torch.no_grad():
            predicted_score = meta_learner(param_tensor).item()
        print(f"Meta-learner predicted score: {predicted_score:.4f}")
        
        # Train model
        start_time = time.time()
        score, accuracy = train_model_with_params(params)
        end_time = time.time()
        
        # Update bandit
        bandit.update(params, score)
        
        # Update meta-learner
        target_score = torch.tensor(score, dtype=torch.float32)
        meta_optimizer.zero_grad()
        predicted_score = meta_learner(param_tensor)
        meta_loss = F.mse_loss(predicted_score, target_score.view(-1, 1))
        meta_loss.backward()
        meta_optimizer.step()
        
        # Track results
        all_results.append({
            'phase': 'bandit',
            'iteration': i + 1,
            'params': params,
            'score': -score,  # Convert back to loss
            'accuracy': accuracy,
            'time': end_time - start_time
        })
        
        # Update best parameters
        if score > best_score:
            best_params, best_score, best_accuracy = params, score, accuracy
        
        print(f"Current best params: {best_params} with accuracy: {best_accuracy:.2f}%")
    
    # Plot results
    plot_results(all_results)
    
    return best_params, best_score, best_accuracy

# Helper function to plot results
def plot_results(results):
    plt.figure(figsize=(15, 10))
    
    # Extract genetic results
    genetic_results = [r for r in results if r['phase'] == 'genetic']
    genetic_generations = [r['generation'] for r in genetic_results]
    genetic_accuracies = [r['accuracy'] for r in genetic_results]
    genetic_losses = [r['score'] for r in genetic_results]
    
    # Extract bandit results
    bandit_results = [r for r in results if r['phase'] == 'bandit']
    if bandit_results:
        bandit_iterations = [r['iteration'] for r in bandit_results]
        bandit_accuracies = [r['accuracy'] for r in bandit_results]
        bandit_losses = [r['score'] for r in bandit_results]
    
    # Combined timeline for all evaluations
    all_evals = list(range(1, len(results) + 1))
    all_accuracies = [r['accuracy'] for r in results]
    all_losses = [r['score'] for r in results]
    
    # Plot overall progress
    plt.subplot(3, 2, 1)
    plt.plot(all_evals, all_accuracies, 'bo-')
    plt.title('Overall Accuracy vs Evaluation')
    plt.xlabel('Evaluation Number')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(all_evals, all_losses, 'ro-')
    plt.title('Overall Loss vs Evaluation')
    plt.xlabel('Evaluation Number')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot genetic algorithm results
    plt.subplot(3, 2, 3)
    plt.scatter(genetic_generations, genetic_accuracies)
    plt.title('Genetic Algorithm: Accuracy vs Generation')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.scatter(genetic_generations, genetic_losses)
    plt.title('Genetic Algorithm: Loss vs Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot bandit results if available
    if bandit_results:
        plt.subplot(3, 2, 5)
        plt.plot(bandit_iterations, bandit_accuracies, 'go-')
        plt.title('Bandit: Accuracy vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.subplot(3, 2, 6)
        plt.plot(bandit_iterations, bandit_losses, 'mo-')
        plt.title('Bandit: Loss vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hybrid_tuning_results.png')
    plt.close()

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
    
    print(f"Final Model Performance - Loss: {-final_score:.4f}, Accuracy: {final_accuracy:.2f}%")
