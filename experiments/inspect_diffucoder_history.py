import torch
import sys
from transformers import AutoModel, AutoTokenizer

# Mock torchvision as in the original script
class MockModule:
    def __getattr__(self, name): return MockModule()
    def __call__(self, *args, **kwargs): return MockModule()

sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()

if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()

def inspect_history():
    model_id = "apple/DiffuCoder-7B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    input_text = "def hello_world():\n(terrible_var) = 10\n    print(terrible_var)"
    print(f"Input text: {input_text}")
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    print(f"Input IDs: {input_ids}")

    print("Starting generation...")
    # Using parameters likely to produce history
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1], 
        steps=10, 
        output_history=True,
        return_dict_in_generate=True,
        temperature=0., 
    )

    if hasattr(output, 'history'):
        print(f"History available. Length: {len(output.history)}")
        print(f"Type of first element: {type(output.history[0])}")
        
        # Check first and last element
        if isinstance(output.history[0], torch.Tensor):
            print(f"Step 0 shape: {output.history[0].shape}")
            print(f"Step 0 content: {output.history[0]}")
            
            # last step
            print(f"Last step content: {output.history[-1]}")
            
            # Decode them
            print(f"Step 0 decoded: {tokenizer.decode(output.history[0][0], skip_special_tokens=True)}")
            print(f"Last step decoded: {tokenizer.decode(output.history[-1][0], skip_special_tokens=True)}")
            
    else:
        print("No history attribute found.")
        print(f"Output attributes: {dir(output)}")

if __name__ == "__main__":
    inspect_history()
