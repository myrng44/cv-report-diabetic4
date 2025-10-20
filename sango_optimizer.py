"""
SANGO (Self-Adaptive Northern Goshawk Optimization) Algorithm
Implementation based on the paper's methodology
"""

import numpy as np
from typing import Callable, List, Tuple
import torch
import torch.nn as nn


class SANGOOptimizer:
    """
    Self-Adaptive Northern Goshawk Optimization Algorithm
    Used for optimizing GRU hyperparameters
    """

    def __init__(self,
                 objective_function: Callable,
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 20,
                 max_iterations: int = 30,
                 alpha: float = 2.0,
                 beta: float = 1.5):
        """
        Initialize SANGO optimizer

        Args:
            objective_function: Function to minimize (e.g., validation loss)
            dim: Dimension of search space
            bounds: List of (min, max) tuples for each dimension
            population_size: Number of solutions in population
            max_iterations: Maximum number of iterations
            alpha: Self-adaptive parameter
            beta: Exploration parameter
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta

        # Initialize population
        self.population = self._initialize_population()
        self.fitness = np.zeros(population_size)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []

    def _initialize_population(self) -> np.ndarray:
        """Initialize population randomly within bounds"""
        population = np.zeros((self.population_size, self.dim))
        for i in range(self.dim):
            population[:, i] = np.random.uniform(
                self.bounds[i, 0],
                self.bounds[i, 1],
                self.population_size
            )
        return population

    def _evaluate_fitness(self) -> None:
        """Evaluate fitness for all solutions"""
        for i in range(self.population_size):
            self.fitness[i] = self.objective_function(self.population[i])

            # Update best solution
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()

    def _levy_flight(self, Lambda: float = 1.5) -> np.ndarray:
        """Generate Levy flight random walk"""
        import math

        sigma = (
            math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
            (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2))
        ) ** (1 / Lambda)

        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / np.abs(v) ** (1 / Lambda)

        return step

    def _update_position(self, position: np.ndarray, iteration: int) -> np.ndarray:
        """Update position using Northern Goshawk hunting strategy"""
        # Self-adaptive parameter
        r = self.alpha * (1 - iteration / self.max_iterations)

        # Random parameters
        r1 = np.random.rand()
        r2 = np.random.rand()

        # Choose random solution from population
        random_idx = np.random.randint(0, self.population_size)
        random_solution = self.population[random_idx]

        # Northern Goshawk hunting behavior
        if r1 < 0.5:
            # Exploitation: Attack prey (best solution)
            levy = self._levy_flight()
            new_position = self.best_solution + r * levy * (self.best_solution - position)
        else:
            # Exploration: Search for prey
            new_position = random_solution + self.beta * r2 * (random_solution - position)

        # Apply bounds
        new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

        return new_position

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run SANGO optimization

        Returns:
            best_solution: Optimal parameters found
            best_fitness: Best fitness value
        """
        # Initial evaluation
        self._evaluate_fitness()
        self.convergence_curve.append(self.best_fitness)

        if verbose:
            print(f"Iteration 0: Best Fitness = {self.best_fitness:.6f}")

        # Main optimization loop
        for iteration in range(1, self.max_iterations + 1):
            # Update each solution
            for i in range(self.population_size):
                new_position = self._update_position(self.population[i], iteration)
                new_fitness = self.objective_function(new_position)

                # Greedy selection
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_position.copy()

            self.convergence_curve.append(self.best_fitness)

            if verbose and iteration % 5 == 0:
                print(f"Iteration {iteration}: Best Fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness


class OptimizedGRU(nn.Module):
    """
    Optimized GRU with parameters tuned by SANGO
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_classes: int = 5):
        super(OptimizedGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Bidirectional for better feature learning
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # GRU forward
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size*2)

        # Attention weights
        attention_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention
        context = torch.sum(gru_out * attention_weights, dim=1)  # (batch, hidden_size*2)

        # Classification
        output = self.classifier(context)  # (batch, num_classes)

        return output


def create_optimized_gru(input_size: int,
                        num_classes: int = 5,
                        hidden_size: int = 128,
                        num_layers: int = 2,
                        dropout: float = 0.3) -> OptimizedGRU:
    """Factory function to create OptimizedGRU model"""
    return OptimizedGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    )
