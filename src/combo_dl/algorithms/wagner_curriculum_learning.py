from collections.abc import Sequence
from typing import Any, override

from combo_dl import models
import combo_dl.algorithms as algs
from combo_dl.experiment_logger.experiment_logger import ExperimentLogger
import combo_dl.problems as probs


class DCECurriculumLearning(algs.BaseAlgorithm):
    def __init__(
        self,
        model: models.SamplingModel,
        problems: Sequence[probs.StronglyRegularGraphs],
        logger: ExperimentLogger,
        device: str = "cpu",
    ):
        self.model = model
        self.problems = problems
        self.logger = ExperimentLogger(wandb_mode="disabled")
        self.device = device

    @override
    def optimize(self, **kwargs) -> dict[str, Any]:
        for problem in self.problems:
            dce_run = algs.WagnerDeepCrossEntropy(
                self.model,
                problem,
                iterations=100_000,
                batch_size=1028,
                logger=self.logger,
                device=self.device,
                curriculum_size=problem.n,
            )
            _result = dce_run.optimize()

        return {}


if __name__ == "__main__":
    parameters = [(5, 2, 0, 1), (9, 4, 1, 2), (10, 3, 0, 1), (13, 6, 2, 3)]
    problems = [
        probs.StronglyRegularGraphs(n, k, lambda_param, mu)
        for (n, k, lambda_param, mu) in parameters
    ]
    model = models.PaddedFFModel(
        n=parameters[-1][0], hidden_layer_sizes=[128, 64, 32, 16, 8, 4], output_size=2
    )
    alg = DCECurriculumLearning(
        model, problems, ExperimentLogger(wandb_mode="disabled"), device="mps"
    )
    alg.optimize()
