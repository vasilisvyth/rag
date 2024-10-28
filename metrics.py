from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCaseParams
import getpass

# see https://arxiv.org/abs/2303.16634
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o-mini",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=False
)