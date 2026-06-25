import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline
# Add external_repos/LLaDA to path for generate import
project_root = os.path.dirname(os.path.abspath(__file__))
llada_path = os.path.join(project_root, 'external_repos', 'LLaDA')
if os.path.exists(llada_path) and llada_path not in sys.path:
    sys.path.append(llada_path)

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

llada_generate = None
try:
    import importlib.util
    gen_path = os.path.join(llada_path, 'generate.py')
    if os.path.exists(gen_path):
        spec = importlib.util.spec_from_file_location("llada_generate_mod", gen_path)
        llada_gen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llada_gen_mod)
        llada_generate = llada_gen_mod.generate
    else:
        # Fallback to simple import if for some reason abspath failed
        import generate as llada_gen_mod
        llada_generate = llada_gen_mod.generate
except Exception as e:
    print(f"Warning: Failed to import llada_generate: {e}")
    pass

class BaseModel:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self):
        raise NotImplementedError

    def generate(self, prompt, **kwargs):
        raise NotImplementedError

class CodeGemmaModel(BaseModel):
    def load(self):
        if LLM is None:
            raise ImportError("vllm is not installed")
        # CodeGemma usually works with auto configuration in vLLM
        self.model = LLM(model=self.model_id)

    def generate(self, prompt, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded. Did you call .load()?")
        sampling_params = SamplingParams(max_tokens=kwargs.get('max_tokens', 8192))
        messages = [{"role": "user", "content": prompt}]
        outputs = self.model.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

class DeepSeekModel(BaseModel):
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # Use max_new_tokens instead of max_length to avoid error with long inputs
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=kwargs.get('max_new_tokens', 128),
            use_cache=kwargs.get('use_cache', False)
        )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

class DiffuCoderModel(BaseModel):
    def load(self):
        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def generate(self, prompt, **kwargs):
        is_infill = kwargs.get('is_infill', False)
        mask_token = kwargs.get('mask_token', '<|mask|>')
        
        if is_infill:
            # For infilling, we replace [MASK] with four mask tokens
            # to allow more space for identifier generation
            num_masks = kwargs.get('num_masks', 4)
            inference_text = prompt.replace('[MASK]', mask_token * num_masks)
            inputs = self.tokenizer(inference_text, return_tensors="pt")
            input_ids = inputs.input_ids.to(device="cuda")
            attention_mask = inputs.attention_mask.to(device="cuda")
            
            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1, 
                steps=kwargs.get('steps', 256),
                temperature=kwargs.get('temperature', 0.3),
                top_p=0.95,
                alg="entropy",
                alg_temp=0.,
            )
            seqs = output.sequences if hasattr(output, "sequences") else output
            decoded = self.tokenizer.decode(seqs[0], skip_special_tokens=True)
            # Find the part that was masked. This is tricky. 
            # Often we just return the whole denoised code if it's for refactoring.
            return decoded
        else:
            full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt.strip()}\n<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device="cuda")
            attention_mask = inputs.attention_mask.to(device="cuda")

            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get('max_new_tokens', 256),
                steps=kwargs.get('steps', 256),
                temperature=kwargs.get('temperature', 0.3),
                top_p=0.95,
                alg="entropy",
                alg_temp=0.,
            )
            seqs = output.sequences if hasattr(output, "sequences") else output
            generations = [self.tokenizer.decode(g[len(p):].tolist()) for p, g in zip(input_ids, seqs)]
            return generations[0].split('<|dlm_pad|>')[0]

class LLaDAModel(BaseModel):
    def load(self):
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def generate(self, prompt, **kwargs):
        if llada_generate is None:
            raise ImportError("Could not import generate from LLaDA repo")
        
        is_infill = kwargs.get('is_infill', False)
        mask_id = 126336 # Standard LLaDA mask_id

        if is_infill:
            # Replace [MASK] with four mask tokens
            num_masks = kwargs.get('num_masks', 4)
            m_text = self.tokenizer.decode([mask_id])
            input_text = prompt.replace('[MASK]', m_text * num_masks)
            inputs = self.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs.input_ids.to('cuda')
            
            # Ensure mask tokens are correctly identified as mask_id in input_ids
            # (Sometimes tokenizer encodes the text representation differently)
            m_encoded = self.tokenizer.encode(m_text, add_special_tokens=False)
            for m_id in m_encoded:
                input_ids[input_ids == m_id] = mask_id

            out = llada_generate(
                self.model, 
                input_ids, 
                steps=kwargs.get('steps', 128), 
                gen_length=kwargs.get('gen_length', 1), # 1 triggers infilling mode in some setups, or we use the whole len
                block_length=kwargs.get('block_length', 32),
                mask_id=mask_id
            )
            # The output will have the mask filled
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            m = [{"role": "user", "content": prompt}]
            user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = self.tokenizer(user_input)['input_ids']
            input_ids = torch.tensor(input_ids).to('cuda').unsqueeze(0)
            
            gen_length = kwargs.get('gen_length', 128)
            steps = kwargs.get('steps', 128)
            
            out = llada_generate(self.model, input_ids, steps=steps, gen_length=gen_length, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
            answer = self.tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            return answer

class LlamaModel(BaseModel):
    def load(self):
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def generate(self, prompt, **kwargs):
        try:
            messages = [
                {"role": "system", "content": kwargs.get('system_prompt', "You are a helpful assistant.")},
                {"role": "user", "content": prompt},
            ]
            outputs = self.pipeline(messages, max_new_tokens=kwargs.get('max_new_tokens', 256))
            return outputs[0]["generated_text"][-1]['content']
        except Exception as e:
            if "gated repo" in str(e):
                return "Error: This model is gated. Please login using `huggingface-cli login` with a token that has access to Llama 3."
            raise e

class QwenModel(BaseModel):
    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def generate(self, prompt, **kwargs):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=kwargs.get('max_new_tokens', 512))
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

class DreamCoderModel(BaseModel):
    def load(self):
        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def generate(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        input_ids = inputs.input_ids.to(device="cuda")
        attention_mask = inputs.attention_mask.to(device="cuda")

        output = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=kwargs.get('max_new_tokens', 768),
            steps=kwargs.get('steps', 768),
            temperature=kwargs.get('temperature', 0.1),
            top_p=0.95,
            alg="entropy",
            alg_temp=0.,
        )
        
        # Handle cases where output is a Tensor or a dict-like object
        if hasattr(output, "sequences"):
            seqs = output.sequences
        else:
            seqs = output

        generations = [self.tokenizer.decode(g[len(p):].tolist()) for p, g in zip(input_ids, seqs)]
        return generations[0].split(self.tokenizer.eos_token)[0]

MODEL_REGISTRY = {
    "codegemma": {"class": CodeGemmaModel, "id": "google/codegemma-7b-it"},
    "deepseek": {"class": DeepSeekModel, "id": "deepseek-ai/deepseek-coder-6.7b-instruct"},
    "diffucoder": {"class": DiffuCoderModel, "id": "apple/DiffuCoder-7B-Instruct"},
    "llada": {"class": LLaDAModel, "id": "GSAI-ML/LLaDA-8B-Instruct"},
    "llama": {"class": LlamaModel, "id": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    "qwen": {"class": QwenModel, "id": "Qwen/Qwen2.5-Coder-7B-Instruct"},
    "dreamcoder": {"class": DreamCoderModel, "id": "Dream-org/Dream-Coder-v0-Instruct-7B"},
}

def run_model(model_key, prompt, **kwargs):
    if model_key not in MODEL_REGISTRY:
        print(f"Model {model_key} not found. Available: {list(MODEL_REGISTRY.keys())}")
        return

    config = MODEL_REGISTRY[model_key]
    model_instance = config["class"](config["id"])
    print(f"Loading {model_key}...")
    model_instance.load()
    print(f"Generating with {model_key}...")
    response = model_instance.generate(prompt, **kwargs)
    return response

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.prog = "python unified_framework.py"
    parser.add_argument("--model", type=str, required=True, help="Model name or 'all'")
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        print(f"\n{'='*20} RUNNING {model_key.upper()} {'='*20}")
        try:
            result = run_model(model_key, args.prompt)
            print("\n" + "-"*10 + " RESPONSE " + "-"*10)
            print(result)
        except Exception as e:
            print(f"Error running {model_key}: {e}")
        finally:
            import gc
            # Cleanup efforts to prevent OOM when running multiple models
            if 'result' in locals(): del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        print("="*50)
