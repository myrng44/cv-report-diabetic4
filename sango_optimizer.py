"""
SANGO (Tối ưu hóa Northern Goshawk Tự thích nghi)
Triển khai dựa trên phương pháp luận của bài báo
"""

import numpy as np
from typing import Callable, List, Tuple
import torch
import torch.nn as nn


class SANGOOptimizer:
    """
    Thuật toán Tối ưu hóa Northern Goshawk Tự thích nghi
    Sử dụng để tối ưu hóa siêu tham số GRU
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
        Khởi tạo bộ tối ưu SANGO

        Args:
            objective_function: Hàm để tối thiểu hóa (vd: validation loss)
            dim: Số chiều của không gian tìm kiếm
            bounds: Danh sách các tuple (min, max) cho mỗi chiều
            population_size: Số lượng giải pháp trong quần thể
            max_iterations: Số lần lặp tối đa
            alpha: Tham số tự thích nghi
            beta: Tham số khám phá
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta

        # Khởi tạo quần thể
        self.population = self._initialize_population()
        self.fitness = np.zeros(population_size)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []

    def _initialize_population(self) -> np.ndarray:
        """Khởi tạo quần thể ngẫu nhiên trong giới hạn"""
        population = np.zeros((self.population_size, self.dim))
        for i in range(self.dim):
            population[:, i] = np.random.uniform(
                self.bounds[i, 0],
                self.bounds[i, 1],
                self.population_size
            )
        return population

    def _evaluate_fitness(self) -> None:
        """Đánh giá fitness cho tất cả các giải pháp"""
        for i in range(self.population_size):
            self.fitness[i] = self.objective_function(self.population[i])

            # Cập nhật giải pháp tốt nhất
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()

    def _levy_flight(self, Lambda: float = 1.5) -> np.ndarray:
        """Tạo bước đi ngẫu nhiên Levy flight"""
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
        """Cập nhật vị trí sử dụng chiến lược săn mồi Northern Goshawk"""
        # Tham số tự thích nghi
        r = self.alpha * (1 - iteration / self.max_iterations)

        # Tham số ngẫu nhiên
        r1 = np.random.rand()
        r2 = np.random.rand()

        # Chọn giải pháp ngẫu nhiên từ quần thể
        random_idx = np.random.randint(0, self.population_size)
        random_solution = self.population[random_idx]

        # Hành vi săn mồi của Northern Goshawk
        if r1 < 0.5:
            # Khai thác: Tấn công con mồi (giải pháp tốt nhất)
            levy = self._levy_flight()
            new_position = self.best_solution + r * levy * (self.best_solution - position)
        else:
            # Khám phá: Tìm kiếm con mồi
            new_position = random_solution + self.beta * r2 * (random_solution - position)

        # Áp dụng giới hạn
        new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

        return new_position

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Chạy tối ưu hóa SANGO

        Returns:
            best_solution: Tham số tối ưu tìm được
            best_fitness: Giá trị fitness tốt nhất
        """
        # Đánh giá ban đầu
        self._evaluate_fitness()
        self.convergence_curve.append(self.best_fitness)

        if verbose:
            print(f"Iteration 0: Best Fitness = {self.best_fitness:.6f}")

        # Vòng lặp tối ưu hóa chính
        for iteration in range(1, self.max_iterations + 1):
            # Cập nhật từng giải pháp
            for i in range(self.population_size):
                new_position = self._update_position(self.population[i], iteration)
                new_fitness = self.objective_function(new_position)

                # Lựa chọn tham lam
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness

                    # Cập nhật giải pháp tốt nhất toàn cục
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_position.copy()

            self.convergence_curve.append(self.best_fitness)

            if verbose and iteration % 5 == 0:
                print(f"Iteration {iteration}: Best Fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness


class OptimizedGRU(nn.Module):
    """
    GRU tối ưu hóa với tham số được điều chỉnh bởi SANGO
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

        # Các lớp GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Hai chiều để học đặc trưng tốt hơn
        )

        # Cơ chế attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Đầu phân loại
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến với cơ chế attention

        Args:
            x: Tensor đầu vào có shape (batch, seq_len, input_size)

        Returns:
            Logits đầu ra có shape (batch, num_classes)
        """
        # GRU tiến
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size*2)

        # Trọng số attention
        attention_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Áp dụng attention
        context = torch.sum(gru_out * attention_weights, dim=1)  # (batch, hidden_size*2)

        # Phân loại
        output = self.classifier(context)  # (batch, num_classes)

        return output


def create_optimized_gru(input_size: int,
                        num_classes: int = 5,
                        hidden_size: int = 128,
                        num_layers: int = 2,
                        dropout: float = 0.3) -> OptimizedGRU:
    """Hàm factory để tạo mô hình OptimizedGRU"""
    return OptimizedGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    )
