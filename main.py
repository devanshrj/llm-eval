import argparse
import logging
import os

from evaluation import HumanEval
from models import GeminiLLM


DATASETS = {
    "humaneval": HumanEval,
}


MODELS = {
    "gemini-pro": GeminiLLM,
}


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model to evaluate",
        default="gemini-pro",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to evaluate on",
        default="humaneval",
    )
    parser.add_argument(
        "--key",
        type=str,
        help="API key for the model if using a provider",
        required=True,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Dataset path",
        default='human-eval/data/HumanEval.jsonl.gz',
    )
    parser.add_argument(
        "--out-path",
        type=str,
        help="Output path",
        default='results',
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of samples per task",
        default=3,
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the evaluation.
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("Retrieving command line arguments")
    args = parse_args()
    logging.info(args)

    # retrieve args
    model_name = args.model
    api_key = args.key
    n_sample = args.n
    data_path = args.data_path
    out_path = args.out_path
    os.makedirs(args.out_path, exist_ok=True)

    # create model and dataset objects
    logging.info("Creating model and dataset objects")
    llm = MODELS[model_name](model_name, api_key=api_key)
    evaluator = DATASETS[args.dataset]()

    # run evaluation
    logging.info("Running evaluation")
    result = evaluator.evaluate(llm, data_path, out_path, n_sample=n_sample)
    print(result)


if __name__ == "__main__":
    main()
