import os
from abc import ABC, abstractmethod
from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness as he_evaluate

from humaneval_x.utils import read_dataset
from humaneval_x.evaluation import evaluate_functional_correctness as hex_evaluate

from models import BaseLLM

import logging
log = logging.getLogger(__name__)


class Evaluator(ABC):
    """
    Base class to create wrappers for evalauting on different datasets.
    """
    @abstractmethod
    def evaluate(self, model: BaseLLM, data_path: str, out_path: str, **kwargs) -> dict:
        """
        Evaluate the given model on the dataset data_path and write the results to out_path.
        """
        pass


class HumanEval(Evaluator):
    """
    Wrapper for evaluating on the HumanEval dataset.
    """

    def entry_point(
        self,
        problem_file: str,
        sample_file: str,
        k: str = "1,10,100",
        n_workers: int = 4,
        timeout: float = 3.0,
    ):
        """
        Evaluates the functional correctness of generated samples, and writes
        results to f"{sample_file}_results.jsonl.gz"
        """
        k = list(map(int, k.split(",")))
        results = he_evaluate(
            sample_file, k, n_workers, timeout, problem_file
        )

        return results

    def create_prompt(self, prompt: str) -> str:
        """
        Create an instruction-based prompt for the given problem.
        """
        PROMPT_TEMPLATE = """Please generate code to complete the following problem.\n```python\n{prompt}\n```\nEnclose your solution in triple backticks."""
        return PROMPT_TEMPLATE.format(prompt=prompt)

    def process_code(self, completion: str) -> str:
        """
        Post-process generated code to remove unnecessary parts.
        Based on: https://github.com/abacaj/code-eval/blob/main/process_eval.py
        """
        completion = completion.replace("\r", "")
        if '```python' in completion:
            def_line = completion.index('```python')
            completion = completion[def_line:].strip()
            completion = completion.replace('```python', '')
            try:
                next_line = completion.index('```')
                completion = completion[:next_line].strip()
            except:
                log.error(completion)
        if "__name__ == \"__main__\"" in completion:
            next_line = completion.index('if __name__ == "__main__":')
            completion = completion[:next_line].strip()
        if "# Example usage" in completion:
            next_line = completion.index('# Example usage')
            completion = completion[:next_line].strip()

        return completion

    def evaluate(self, model: BaseLLM, data_path: str, out_path: str, **kwargs) -> dict:
        log.info("Reading problems...")
        dataset = read_problems(data_path)
        n_sample = kwargs.get("n_sample", 1)
        use_template = kwargs.get("use_template", True)
        post_process = kwargs.get("post_process", True)
        samples = []

        log.info("Generating samples...")
        progress_bar = tqdm(total=len(dataset) * n_sample,
                            desc="Generating samples")

        for task_id in dataset:
            prompt = dataset[task_id]["prompt"]
            if use_template:
                prompt = self.create_prompt(prompt)
            completions = model.generate_completions(prompt, n_sample=n_sample)
            for completion in completions:
                if post_process:
                    completion = self.process_code(completion)
                sample = dict(task_id=task_id,
                              completion=completion)
                samples.append(sample)
            progress_bar.update(n_sample)
        progress_bar.close()

        log.info("Storing samples...")
        model_name = model.model_name.replace("/", "_")
        pred_filename = f"{out_path}/{model_name}_predictions.jsonl"
        write_jsonl(pred_filename, samples)

        log.info("Evaluating samples...")
        result = self.entry_point(
            problem_file=data_path, sample_file=pred_filename)
        return result

class HumanEvalX(Evaluator):
    def create_prompt(self, prompt: str, language: str) -> str:
        """
        Create an instruction-based prompt for the given problem.
        """
        PROMPT_TEMPLATE = """Please generate code to complete the following problem.\n```{language}\n{prompt}\n```\nEnclose your solution in triple backticks. Only generate the code block to finish the code, do not include the prompt, function definition or any explanation."""
        return PROMPT_TEMPLATE.format(prompt=prompt, language=language)
    
    def process_code(self, completion: str, language: str) -> str:
        """
        Post-process generated code to remove unnecessary parts.
        Based on: https://github.com/abacaj/code-eval/blob/main/process_eval.py
        """
        completion = completion.replace("\r", "")
        if f'```{language}' in completion:
            def_line = completion.index(f'```{language}')
            completion = completion[def_line:].strip()
            completion = completion.replace(f'```{language}', '')
            try:
                next_line = completion.index('```')
                completion = completion[:next_line]
            except:
                log.error(completion)
        if "__name__ == \"__main__\"" in completion:
            next_line = completion.index('if __name__ == "__main__":')
            completion = completion[:next_line].strip()
        if "# Example usage" in completion:
            next_line = completion.index('# Example usage')
            completion = completion[:next_line].strip()

        return completion
    
    def evaluate(self, model: BaseLLM, data_path: str, out_path: str, **kwargs) -> dict:
        log.info("Reading problems...")
        dataset = read_dataset(data_path, dataset_type="humaneval")
        n_sample = kwargs.get("n_sample", 1)
        use_template = kwargs.get("use_template", True)
        post_process = kwargs.get("post_process", True)
        samples = []

        log.info("Generating samples...")
        progress_bar = tqdm(total=len(dataset) * n_sample,
                            desc="Generating samples")

        for task_id in dataset:
            language = task_id.split("/")[0].lower()
            og_prompt = dataset[task_id]["prompt"]
            if use_template:
                prompt = self.create_prompt(og_prompt, language)
            completions = model.generate_completions(prompt, n_sample=n_sample)
            for completion in completions:
                if post_process:
                    completion = self.process_code(completion, language)
                sample = dict(task_id=task_id,
                              generation=completion,
                              prompt=og_prompt)
                samples.append(sample)
            progress_bar.update(n_sample)
        progress_bar.close()

        log.info("Storing samples...")
        model_name = model.model_name.replace("/", "_")
        pred_filename = f"{out_path}/{model_name}_predictions.jsonl"
        write_jsonl(pred_filename, samples)

        log.info("Evaluating samples...")
        tmp_path = f"{os.getcwd()}/executions/{model_name}"
        hex_evaluate(
            input_file=pred_filename,
            problem_file=data_path,
            out_dir=out_path,
            tmp_dir=tmp_path,
        )
        log.info("Evaluation complete.")