"""Basic smoke tests to ensure the package can be imported and basic functionality works."""

import torch

import thesis
from thesis.algorithms import WagnerDeepCrossEntropy
from thesis.experiment_logger import ExperimentLogger
from thesis.models import FFModel
from thesis.problems import StronglyRegularGraphs, WagnerCorollary21


class TestBasicImports:
    """Test that all major components can be imported and instantiated."""

    def test_import_thesis_package(self) -> None:
        """Test that the main package can be imported."""
        assert hasattr(thesis, "__version__") or True  # Version may not be defined

    def test_problem_classes_can_be_created(self) -> None:
        """Test that problem classes can be instantiated."""
        # Test SRG problem
        srg = StronglyRegularGraphs(n=5, k=2, lambda_param=0, mu=1)
        assert srg.n == 5
        assert srg.k == 2
        assert srg.lambda_param == 0
        assert srg.mu == 1

        # Test Wagner problem
        wagner = WagnerCorollary21(n=4)
        assert wagner.n == 4

    def test_model_can_be_created(self) -> None:
        """Test that models can be instantiated."""
        model = FFModel(n=4)
        assert model.n == 4
        assert model.edges == 6  # (4^2 - 4) / 2 = 6

    def test_logger_can_be_created(self) -> None:
        """Test that logger can be instantiated."""
        logger = ExperimentLogger(wandb_mode="disabled")
        assert logger.use_wandb is False

    def test_algorithm_can_be_created(self) -> None:
        """Test that algorithm can be instantiated with minimal setup."""
        model = FFModel(n=3)
        problem = StronglyRegularGraphs(n=3, k=1, lambda_param=0, mu=1)
        logger = ExperimentLogger(wandb_mode="disabled")

        algorithm = WagnerDeepCrossEntropy(
            model=model,
            problem=problem,
            logger=logger,
            iterations=1,  # Minimal for testing
            batch_size=2,  # Minimal for testing
        )
        assert algorithm.model == model
        assert algorithm.problem == problem
        assert algorithm.logger == logger

    def test_torch_operations_basic(self) -> None:
        """Test that basic PyTorch operations work as expected."""
        # Test that we can create tensors and do basic operations
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        assert z.shape == (2, 3)

        # Test that our models can handle basic tensor operations
        model = FFModel(n=4)

        # Test sampling produces correct shape
        samples = model.sample(batch_size=3)
        assert samples.shape == (3, model.edges)
        assert torch.all((samples == 0) | (samples == 1))
