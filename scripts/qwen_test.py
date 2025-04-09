import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from moshi.utils.sampling import sample_token

import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

def load_qwen_model():
    # Path to where the model weights are stored
    weights_dir = "weights/qwen"
    
    # Load the tokenizer
    print("Loading Qwen2.5-3B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    
    # Load the model from the safetensor files
    # Setting device_map to "auto" will distribute the model across available GPUs
    # or load to CPU if no GPUs are available
    print("Loading Qwen2.5-3B model...")
    model = AutoModelForCausalLM.from_pretrained(
        weights_dir,
        device_map="auto",
        torch_dtype=torch.float16,  # Use float16 for efficiency
        trust_remote_code=True,  # Needed for some Qwen-specific code
        output_hidden_states=True,  # Enable hidden states output
        output_attentions=True,  # Also enable attentions if needed
        return_dict_in_generate=True  # Return a model output object instead of just tokens
    )
    
    return model, tokenizer

def generate_text_with_states(model, tokenizer, prompts, max_length=100):
    # Tokenize the prompts in batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    # Generate responses with additional outputs
    print(f"Shape of input: {inputs.input_ids.shape}")
    print(f"Generating text for {len(prompts)} prompts...")
    output = model(inputs.input_ids, output_hidden_states=True, output_attentions=True)
    # print keys of output
    print(f"Output keys: {output.keys()}")
    # output - CausalLMOutputWithPast, keys - loss, logits
    print(f"Shape of output: {output.logits.shape}")
    print(f"Length of hidden states: {len(output.hidden_states)}")
    print(f"Shape of hidden states: {output.hidden_states[-1].shape}")

    generation_output = model.generate(
        inputs.input_ids,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_dict_in_generate=True,  # This ensures we get a ModelOutput object
        output_scores=True,  # Return scores (logits after softmax)
        output_hidden_states=True,  # Return hidden states
        output_attentions=True  # Return attention weights
    )
    
    # Process results for each item in the batch
    results = []
    
    # Extract the generated tokens (now a batch)
    generated_tokens = generation_output.sequences
    
    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    if generation_output.hidden_states:
        last_hidden_states = [hidden_state[-1] for hidden_state in generation_output.hidden_states]
        # Stack along the second dimension (dim=1) to get B, T, D
        final_hidden_states = torch.cat(last_hidden_states, dim=1)
    else:
        final_hidden_states = None

    result = {
        "texts": generated_texts,
        "tokens": generated_tokens,
        "logits": generation_output.scores if generation_output.scores else None,
        "hidden_states": final_hidden_states
    }
    
    # Return the single result dictionary
    return result

def batch_generate_example(model, tokenizer, prompts, max_length=100):
    result = generate_text_with_states(model, tokenizer, prompts, max_length)
    
    # Print results for each prompt
    for i, (prompt, generated_text) in enumerate(zip(prompts, result["texts"])):
        print(f"\n--- Result {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Generated text: {generated_text}")
    
    if result["hidden_states"] is not None:
        print(f"Shape of hidden states: {result['hidden_states'].shape}")
        
        # Extract last token hidden state for the first item in batch
        last_token_hidden_state = result["hidden_states"][0, -1]
        print(f"Final hidden state for last token shape: {last_token_hidden_state.shape}")
    
    return result

def concatanation(tensor):
    B,T,D = tensor.shape
    rand_hell_tensor = torch.rand(B,T,4096).to(tensor.device)
    combined = torch.cat([tensor, rand_hell_tensor], dim=2)
    return combined

if __name__ == "__main__":
    # Load the model and tokenizer
    model, tokenizer = load_qwen_model()
    
    # Example of batch generation
    print("\n=== Batch Generation Example ===")
    batch_prompts = [
        "Write a short poem about artificial intelligence:",
        "Explain the concept of neural networks in simple terms:"
    ]
    
    batch_result = batch_generate_example(model, tokenizer, batch_prompts)
    
    # Example of how to compute similarity between generated embeddings
    if batch_result["hidden_states"] is not None:
        # Get the average hidden state (embedding) for different items in the batch
        avg_hidden_state_1 = torch.mean(batch_result["hidden_states"][0:1], dim=1)
        avg_hidden_state_2 = torch.mean(batch_result["hidden_states"][1:2], dim=1)
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            avg_hidden_state_1, avg_hidden_state_2
        )
        print(f"\nCosine similarity between the two generated texts: {cos_sim.item():.4f}")

    # Test of how it might work in Moshi
    print("\n=== Moshi Example ===")
    concatanated_tensor = concatanation(batch_result["hidden_states"])
    projector = MLPProjector(2048+4096, 4096, 2048).to(concatanated_tensor.device)

    projected_tensor = projector(concatanated_tensor).to(torch.float16)
    print(f"Projected tensor shape: {projected_tensor.shape}")

    lm_head = model.get_output_embeddings()
    output_logits = lm_head(projected_tensor)
    print(f"Output logits shape: {output_logits.shape}")

    text_tokens = sample_token(output_logits.float(), True, 0.7, 25)
    print(f"Sampled tokens shape: {text_tokens.shape}")