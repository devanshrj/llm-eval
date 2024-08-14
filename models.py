from abc import ABC, abstractmethod
import google.generativeai as genai
import litellm

import dotenv
dotenv.load_dotenv()


class BaseLLM(ABC):
    """
    Base class to create wrappers for large language models from different providers.
    """
    @abstractmethod
    def generate_completions(self, input_prompt: str, **kwargs) -> list[str]:
        """
        Generate num_samples number of completions for a given input prompt.
        """
        pass


class GeminiLLM(BaseLLM):
    """
    Wrapper for Gemini language models.
    """

    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initialize a Gemini model with the given API key.
        """
        api_key = kwargs['api_key']
        genai.configure(api_key=api_key)
        self.model_path = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate_completions(self, input_prompt: str, **kwargs) -> list[str]:
        sample_completions = []
        n_sample = kwargs.get("n_sample", 1)
        for _ in range(n_sample):
            response = self.model.generate_content(input_prompt)
            # response.text sometimes throws a ValueError because model output is stored in response.parts instead
            try:
                completion = response.text
            except ValueError:
                completion = response.parts
            # 'completion': [] is returned when the model fails to generate a completion -> breaks JSON formatting
            completion = str(completion)
            sample_completions.append(completion)
        return sample_completions


ND2LITELLM = {
    # openai
    "openai/gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
    "openai/gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
    "openai/gpt-4-turbo-2024-04-09": "gpt-4-turbo-2024-04-09",
    "openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
    # anthropic
    "anthropic/claude-3-opus-20240229": "claude-3-opus-20240229",
    "anthropic/claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
    # google
    "google/gemini-1.5-pro-latest": "gemini/gemini-1.5-pro-latest",
    # togetherai
    "togetherai/Qwen2-72B-Instruct": "together_ai/Qwen/Qwen2-72B-Instruct",
    "togetherai/llama-3.1-405b-instruct": "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "togetherai/llama-3.1-70b-instruct": "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "togetherai/llama-3.1-8b-instruct": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    # # fireworksai
    "fireworksai/llama-3.1-405b-instruct": "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct",
    "fireworksai/llama-3.1-70b-instruct": "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",
    # mistral
    "mistral/mistral-large-2407": "mistral/mistral-large-2407",
}

class LLM(BaseLLM):
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = ND2LITELLM[model_name]
        self.temperature = kwargs.get("temperature", 0.0)
        self.n = kwargs.get("n", 1)
    
    def generate_completions(self, input_prompt: str, **kwargs) -> list[str]:
        messages = [{ "content": input_prompt, "role": "user"}]
        sample_completions = []
        n_sample = kwargs.get("n_sample", 1)
        for _ in range(n_sample):
            response = litellm.completion(model=self.model_name, messages=messages, temperature=self.temperature)
            sample_completions.append(response.choices[0].message.content)
        return sample_completions