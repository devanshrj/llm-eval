from abc import ABC, abstractmethod
import google.generativeai as genai


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
