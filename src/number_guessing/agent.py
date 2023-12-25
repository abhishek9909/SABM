import time
import os
import openai
from huggingface_hub import InferenceClient

class Agent:
    def __init__(self, temperature=1.0, model='gpt-3.5-turbo', max_tokens=200, api_key = '', api_type = '', api_base = ''):
        if api_type == "openai":
            self.api_type = api_type
            self.temperature = temperature
            self.model = model
            self.max_tokens = max_tokens
            openai.api_key = api_key
            self.client = openai.OpenAI()
        elif api_type == "azure_openai":
            self.api_type = api_type
            self.temperature = temperature
            self.model = model
            self.max_tokens = max_tokens
            os.environ['AZURE_OPENAI_API_KEY'] = api_key
            self.client = openai.AzureOpenAI(api_version = "2023-08-01-preview", azure_endpoint = f"https://{api_base}.openai.azure.com")
        elif api_type == "gradio":
            self.api_type = api_type
            self.temperature = temperature
            self.model = model
            self.max_tokens = max_tokens
            self.client = InferenceClient(self.model, token=api_key)
    
    def communicate(self, context):
        prompt = context + "\n\n"
        message = ""

        retries = 3
        backoff_factor = 2
        current_retry = 0

        while current_retry < retries:
            try:
                if self.api_type != "gradio":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": ""}
                        ],
                        max_tokens=self.max_tokens,
                        n=1,
                        temperature=self.temperature,
                        top_p=1
                    )
                    message = response.choices[0].message.content.strip()
                    return message
                else:
                    generate_kwargs = dict(
                        temperature=self.temperature,
                        max_new_tokens=self.max_tokens,
                        top_p=0.95,
                        repetition_penalty=1.0,
                    )
                    formatted_prompt = "<s>"
                    formatted_prompt += f"[INST] {prompt} [/INST]"
                    response = self.client.text_generation(formatted_prompt, **generate_kwargs, stream=False, return_full_text=False)
                    return response
            # except openai.error.RateLimitError as e:
            #     if current_retry < retries - 1:
            #         wait_time = backoff_factor ** current_retry
            #         print(f"RateLimitError: Retrying in {wait_time} seconds...")
            #         time.sleep(wait_time)
            #         current_retry += 1
            #     else:
            #         print(f"Error {e}")
            #         raise e
            # except openai.error.APIError as e:
            #     if current_retry < retries - 1:
            #         wait_time = backoff_factor ** current_retry
            #         print(f"RateLimitError: Retrying in {wait_time} seconds...")
            #         time.sleep(wait_time)
            #         current_retry += 1
            #     else:
            #         raise e
            except Exception as e:
                if current_retry < retries - 1:
                    wait_time = backoff_factor ** current_retry
                    print(f"RateLimitError: Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    current_retry += 1
                else:
                    print(f"Error {e}")
                    raise e
