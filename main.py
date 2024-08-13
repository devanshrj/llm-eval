import argparse
import logging
import os

from evaluation import HumanEval
from models import LLM


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
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
        default=1,
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
    n_sample = args.n
    data_path = args.data_path
    out_path = args.out_path
    os.makedirs(args.out_path, exist_ok=True)

    # create model and dataset objects
    logging.info("Creating model and dataset objects")
    MODELS = [
        # 'anthropic/claude-3-5-sonnet-20240620',
        # 'openai/gpt-4o-2024-05-13',
        # 'google/gemini-1.5-pro-latest',
        # 'openai/gpt-4-turbo-2024-04-09',
        # 'anthropic/claude-3-opus-20240229',
        # "togetherai/Llama-3-70b-chat-hf",
        # "replicate/meta-llama-3-70b-instruct",
        # "togetherai/Qwen2-72B-Instruct",
        # "togetherai/llama-3.1-405b-instruct",
        # "togetherai/llama-3.1-70b-instruct",
        # "togetherai/llama-3.1-8b-instruct",
        # "openai/gpt-4o-mini-2024-07-18",
        # "nvidia_nim/llama-3.1-8b-instruct",
        # "replicate/llama-3.1-405b-instruct",
        # "fireworksai/llama-3.1-405b-instruct",
        # "fireworksai/llama-3.1-70b-instruct",
        "mistral/mistral-large-2407"
    ]

    llm = LLM(model_name=MODELS[0])
    evaluator = HumanEval()

    # run evaluation
    logging.info(f"Running evaluation for {MODELS[0]}")
    result = evaluator.evaluate(llm, data_path, out_path, n_sample=n_sample)
    print(result)


if __name__ == "__main__":
    main()
