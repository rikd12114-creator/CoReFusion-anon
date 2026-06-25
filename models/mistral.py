from vllm import LLM
from vllm.sampling_params import SamplingParams

model_name = "mistralai/Ministral-8B-Instruct-2410"

sampling_params = SamplingParams(max_tokens=8192)

# note that running Ministral 8B on a single GPU requires 24 GB of GPU RAM
# If you want to divide the GPU requirement over multiple devices, please add *e.g.* `tensor_parallel=2`
llm = LLM(model=model_name, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")

prompt = "Do we need to think for 10 seconds to find the answer of 1 + 1?"

messages = [
    {
        "role": "user",
        "content": prompt
    },
]

outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
# You don't need to think for 10 seconds to find the answer to 1 + 1. The answer is 2,
# and you can easily add these two numbers in your mind very quickly without any delay.
