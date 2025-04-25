import argparse
from typing import List, Optional, Tuple, Dict, Union
import torch
import random
import numpy as np
from torch import nn
from moshi.models import loaders, QwenLMGen
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Callable
import neptune
from tqdm import tqdm
from safetensors.torch import save_model

class TensorDictDataset(Dataset):
    """Dataset for loading data from a .pt file containing a list of dicts with 'tensor' keys."""
    
    def __init__(self, 
        pt_file_path: str,
        transform: Optional[List[Callable]] = None,
        context_size: int = 750,
        overlap: float = 0.1,
        is_text_only: bool = False
    ):
        self.pt_file_path = pt_file_path
        self.transform = transform
        self.context_size = context_size
        self.overlap = int(context_size * overlap)
        self.is_text_only = is_text_only
        
        self.data = torch.load(pt_file_path, weights_only=True)
        self.chunk_indeces = self._build_chunk_index()

    def _build_chunk_index(self) -> List[Tuple[int, int]]:
        chunk_indeces = []
        for i, sample in enumerate(self.data):
            tensor = sample['tensor']
            length = tensor.shape[-1]
            if length == 0:
                continue
            step = self.context_size - self.overlap
            for start in range(0, max(1, length - self.overlap), step):
                chunk_indeces.append((i, start))
        return chunk_indeces


    def __len__(self) -> int:
        return len(self.chunk_indeces)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx, start_idx = self.chunk_indeces[idx]
        sample = self.data[sample_idx]
        tensor = sample['tensor']
        chunk = tensor[:, start_idx:start_idx + self.context_size]

        # Pad if needed
        if chunk.shape[-1] < self.context_size:
            pad_size = self.context_size - chunk.shape[-1]
            # TODO should the pad be zeros or something else?
            pad_tensor = torch.zeros((chunk.shape[0], pad_size), dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat((chunk, pad_tensor), dim=-1)

        # Apply transformations to the tensor if any
        if self.transform:
            for t in self.transform:
                chunk = t(chunk)

        return {
            'tensor': chunk,
            'is_text_only': self.is_text_only,
        }
    
class TextTokenMasker:
    """Mask text tokens with a given probability."""
    
    def __init__(self, mask_prob: float = 0.3, mask_token_id: int = 0):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both batched and unbatched inputs
        if x.dim() == 3:  # (B, 17, T)
            batch_size, num_streams, seq_len = x.shape
            # Only mask the first stream (text)
            mask = torch.rand(batch_size, seq_len) < self.mask_prob
            mask = mask.unsqueeze(1)  # (B, 1, T)
            mask_expanded = torch.zeros(batch_size, num_streams, seq_len).bool().to(x.device)
            mask_expanded[:, 0, :] = mask.squeeze(1)  # Apply mask only to text stream
            
            # Create a copy to avoid modifying the original
            x_masked = x.clone()
            x_masked[mask_expanded] = self.mask_token_id
            return x_masked
        
        elif x.dim() == 2:  # (17, T)
            num_streams, seq_len = x.shape
            # Only mask the first stream (text)
            mask = torch.rand(seq_len) < self.mask_prob
            mask_expanded = torch.zeros(num_streams, seq_len).bool().to(x.device)
            mask_expanded[0, :] = mask  # Apply mask only to text stream
            
            # Create a copy to avoid modifying the original
            x_masked = x.clone()
            x_masked[mask_expanded] = self.mask_token_id
            return x_masked
        
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")


class TemporalShifter:
    """Shift the timing between text and audio tokens."""
    def __init__(self, min_shift: float = -0.6, max_shift: float = 0.6, fps: int = 50):
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.fps = fps  # Assuming 50 tokens per second
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift_seconds = random.uniform(self.min_shift, self.max_shift)
        shift_tokens = int(shift_seconds * self.fps)
        
        # Handle both batched and unbatched inputs
        if x.dim() == 3:  # (B, 17, T)
            batch_size, num_streams, seq_len = x.shape
            x_shifted = x.clone()
            
            if shift_tokens > 0:
                # Shift audio streams (1-16) forward relative to text
                x_shifted[:, 1:, shift_tokens:] = x[:, 1:, :-shift_tokens]
                x_shifted[:, 1:, :shift_tokens] = 0  # Pad with zeros
            elif shift_tokens < 0:
                # Shift audio streams backward relative to text
                abs_shift = abs(shift_tokens)
                x_shifted[:, 1:, :-abs_shift] = x[:, 1:, abs_shift:]
                x_shifted[:, 1:, -abs_shift:] = 0  # Pad with zeros
            
            return x_shifted
        
        elif x.dim() == 2:  # (17, T)
            num_streams, seq_len = x.shape
            x_shifted = x.clone()
            
            if shift_tokens > 0:
                # Shift audio streams (1-16) forward relative to text
                x_shifted[1:, shift_tokens:] = x[1:, :-shift_tokens]
                x_shifted[1:, :shift_tokens] = 0  # Pad with zeros
            elif shift_tokens < 0:
                # Shift audio streams backward relative to text
                abs_shift = abs(shift_tokens)
                x_shifted[1:, :-abs_shift] = x[1:, abs_shift:]
                x_shifted[1:, -abs_shift:] = 0  # Pad with zeros
            
            return x_shifted
        
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-weight", type=str, default="weights/new/model.safetensors")
    parser.add_argument("--qwen-weight", type=str, default="weights/qwen/")
    parser.add_argument("-e", "--epoch", default=0, type=int, help="Number of epochs to train")
    parser.add_argument("--steps", default=0, type=int, help="Total number of training steps")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--data", type=str, default="training.pt", help="Path to the training data .pt file")
    parser.add_argument("--text-data", type=str, default="training_text.pt", help="Path to text-only training data")
    parser.add_argument("--bs", type=int, default=1, help="Batch size for training")
    parser.add_argument("--warmup", type=float, default=0.1, help="Ratio of steps for warmup for one epoch")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--save-steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Evaluate model every N steps")
    parser.add_argument("--projector-lr", type=float, default=1e-4, help="Learning rate for Temporal Transformer")
    parser.add_argument("--finetune-lr", type=float, default=5e-6, help="Learning rate for Depth Transformer")
    parser.add_argument("--text-lr-multiplier", type=float, default=0.75, help="Learning rate multiplier for text components in audio batches")
    parser.add_argument("--padding-loss-weight", type=float, default=0.5, help="Loss weight for padding tokens")
    parser.add_argument("--mask-prob", type=float, default=0.3, help="Probability of masking text tokens")
    parser.add_argument("--temporal-shift-min", type=float, default=-0.6, help="Minimum temporal shift in seconds")
    parser.add_argument("--temporal-shift-max", type=float, default=0.6, help="Maximum temporal shift in seconds")
    parser.add_argument("--accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--clip", type=float, default=0.0, help="Maximum norm for gradient clipping")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--context", type=int, default=750, help="Context size for training")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap ratio for context size")
    parser.add_argument("-n", "--neptune", action="store_true", help="Use Neptune for logging")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HuggingFace repo to use (defaults to Moshiko)"
    )
    return parser.parse_args()


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_model(device: str, moshi_weight: str, qwen_path: str):
    print("Loading text tokenizer...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path)

    print("Loading Qwen2.5-3B model...")
    qwen = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        device_map="auto",
        torch_dtype=torch.float16,  # Use float16 for efficiency
        trust_remote_code=True,  # Needed for some Qwen-specific code
        output_hidden_states=True,  # Enable hidden states output
        output_attentions=True,  # Also enable attentions if needed
        return_dict_in_generate=True  # Return a model output object instead of just tokens
    )

    print("Loading LM...")
    lm = loaders.get_qwen_lm(moshi_weight, qwen, device)
    lm_gen = QwenLMGen(lm, temp=0.5, temp_text=0.5)
    
    print("Language model loaded successfully")
    
    return lm_gen, qwen_tokenizer


def create_data_loaders(args):
    # Define transformations
    # audio_transforms = [
    #     TextTokenMasker(mask_prob=args.mask_prob),
    #     TemporalShifter(min_shift=args.temporal_shift_min, max_shift=args.temporal_shift_max)
    # ]
    audio_transforms = None # TODO change

    # Create audio dataset
    audio_dataset = TensorDictDataset(
        pt_file_path=args.data, 
        transform=audio_transforms,
        context_size=args.context,
        overlap=args.overlap,
        is_text_only=False
    )
    
    # Create text-only dataset
    # text_dataset = TensorDictDataset(
    #     pt_file_path=args.text_data,
    #     transform=None,
    #     is_text_only=True
    # )
    text_dataset = None # TODO change
    
    # Create data loaders
    audio_loader = DataLoader(
        dataset=audio_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    # text_loader = DataLoader(
    #     dataset=text_dataset,
    #     batch_size=args.bs,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    # )
    text_loader = None # TODO change
    
    return audio_loader, text_loader


def get_model_parameters(model, parameter_type):
    selected_params = []
    if parameter_type == "projectors":
        # Get parameters for qwen and depth transformer projectors
        keywords = ["text_projector", "depformer_projector", "depformer_text_emb"]
        for n, p in model.named_parameters():
            if any(keyword in n.lower() for keyword in keywords):
                selected_params.append(p)
    elif parameter_type == "finetune":
        # Get parameters for finetuning - Qwen class head, backbone of DT
        # final linear layers of DT, audio embedding layers of DT
        keywords = ["text_head", "depformer_emb", "depformer.layers", ".linears."]
        for n, p in model.named_parameters():
            if any(keyword in n.lower() for keyword in keywords):
                selected_params.append(p)
    return selected_params


def create_optimizers_and_schedulers(model, args, warmup_steps):
    # Get parameters for different components
    for n, p in model.lm_model.named_parameters():
        keywords = ["qwen.model", "transformer.layers", "emb."]
        if any(n.lower().startswith(keyword) for keyword in keywords):
            p.requires_grad = False
            p.data = p.data.to(dtype=torch.float16)
        else:
            p.requires_grad = True
            p.data = p.data.to(dtype=torch.float32)
        # print(f"Parameter {n}: requires_grad={p.requires_grad}, dtype={p.dtype}")

    mem_requires_grad = 0
    mem_non_requires_grad = 0
    for n, p in model.lm_model.named_parameters():
        if p.requires_grad:
            mem_requires_grad += p.numel() * p.element_size()
        else:
            mem_non_requires_grad += p.numel() * p.element_size()
    # print(f"Memory for requires_grad: {mem_requires_grad / 1024**2} MB")
    # print(f"Memory for non-requires_grad: {mem_non_requires_grad / 1024**2} MB")

    projectors_params = get_model_parameters(model, "projectors")
    finetune_params = get_model_parameters(model, "finetune")

    mem_projectors = 0
    mem_finetune = 0
    for p, f in zip(projectors_params, finetune_params):
        mem_projectors += p.numel() * p.element_size()
        mem_finetune += f.numel() * f.element_size()
    # print(f"Memory for projectors: {mem_projectors / 1024**2} MB")
    # print(f"Memory for finetune: {mem_finetune / 1024**2} MB")

    # Inspect parameter details
    # print("=== PROJECTOR PARAMETERS ===")
    # for i, p in enumerate(projectors_params):
    #     print(f"Param {i}: shape={p.shape}, size={p.numel() * p.element_size() / 1024**2:.2f}MB, type={p.dtype}")
    #     if i > 10:  # Print just a sample
    #         print(f"... and {len(projectors_params) - i} more")
    #         break

    # print("\n=== FINETUNE PARAMETERS ===")
    # for i, p in enumerate(finetune_params):
    #     print(f"Param {i}: shape={p.shape}, size={p.numel() * p.element_size() / 1024**2:.2f}MB, type={p.dtype}")
    #     if i > 10:  # Print just a sample
    #         print(f"... and {len(finetune_params) - i} more")
    #         break

    # Count number of parameters
    # print(f"Num projector params: {len(projectors_params)}")
    # print(f"Num finetune params: {len(finetune_params)}")
    
    # Create optimizers
    projector_optimizer = torch.optim.AdamW(projectors_params, lr=args.projector_lr)
    finetune_optimizer = torch.optim.AdamW(finetune_params, lr=args.finetune_lr, foreach=False)
    
    # Create schedulers with warmup
    # First warmup, then cosine decay
    projector_warmup = LinearLR(projector_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    projector_cosine = CosineAnnealingLR(projector_optimizer, T_max=args.steps - warmup_steps)
    projector_scheduler = SequentialLR(
        projector_optimizer, 
        schedulers=[projector_warmup, projector_cosine], 
        milestones=[warmup_steps]
    )
    
    finetune_warmup = LinearLR(finetune_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    finetune_cosine = CosineAnnealingLR(finetune_optimizer, T_max=args.steps - warmup_steps)
    finetune_scheduler = SequentialLR(
        finetune_optimizer, 
        schedulers=[finetune_warmup, finetune_cosine], 
        milestones=[warmup_steps]
    )
    
    optimizers = {
        "projector": projector_optimizer,
        "finetune": finetune_optimizer
       
    }
    
    schedulers = {
        "projector": projector_scheduler,
        "finetune": finetune_scheduler
    }
    
    return optimizers, schedulers


def compute_loss(logits, targets, pad_token_id, padding_weight=0.5, ignore_index=-100):
    """Compute cross-entropy loss with reduced weight for padding tokens."""
    # Get loss function
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    
    # Reshape logits and targets for loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Compute unweighted loss
    loss = criterion(logits_flat, targets_flat)
    
    # Create weights based on whether tokens are padding
    weights = torch.ones_like(targets_flat, dtype=torch.float)
    weights[targets_flat == pad_token_id] = padding_weight
    
    # Apply weights to the loss
    weighted_loss = (loss * weights).sum() / weights.sum()
    
    return weighted_loss


def adjust_learning_rate_for_text(optimizer, multiplier):
    """Adjust learning rate for text embedding and linear layer by multiplying with a factor."""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= multiplier
    
    return optimizer


def save_checkpoint(model, optimizers, schedulers, step, args):
    save_model(model.lm_model, os.path.join(args.save_dir, f"model_step_{step}.safetensors"))
    checkpoint_path = os.path.join(args.save_dir, f"metadata_step_{step}.pt")
    # TODO - issue with key audio as we don't yet use the text_only
    # torch.save({
    #     'step': step,
    #     'optimizers': {
    #         'audio_temporal': optimizers['audio']['temporal'].state_dict(),
    #         'audio_depth': optimizers['audio']['depth'].state_dict(),
    #         'text_only_temporal': optimizers['text_only']['temporal'].state_dict(),
    #         'text_only_depth': optimizers['text_only']['depth'].state_dict(),
    #     },
    #     'schedulers': {
    #         'audio_temporal': schedulers['audio']['temporal'].state_dict(),
    #         'audio_depth': schedulers['audio']['depth'].state_dict(),
    #         'text_only_temporal': schedulers['text_only']['temporal'].state_dict(),
    #         'text_only_depth': schedulers['text_only']['depth'].state_dict(),
    #     }
    # }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def get_next_batch(use_audio, audio_iter, text_iter):
    if use_audio:
        batch = next(audio_iter)
        return batch, "audio"
    else:
        batch = next(text_iter)
        return batch, "text_only"
    
    
def calculate_losses(text_logits, audio_logits, target_seq, batch_type, pad_id, args):
    text_loss = compute_loss(
        text_logits, target_seq[:, 0],
        pad_token_id=pad_id,
        padding_weight=args.padding_loss_weight if batch_type == "audio" else 1.0
    )

    if batch_type == "audio":
        audio_losses = [
            compute_loss(
                audio_logits[:, i - 9],
                target_seq[:, i],
                pad_token_id=0,
                padding_weight=args.padding_loss_weight,
                ignore_index=2048
            )
            for i in range(9, 17)
        ]
        audio_loss = torch.stack(audio_losses).mean()
        return text_loss + audio_loss
    return text_loss

def check_grad_nan(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"WARNING: NaN gradient detected in {name}")
            return True
    return False

def check_param_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"WARNING: NaN parameter detected in {name}")
            return True
    return False

def train(model, dataloaders, optimizers, schedulers, pad_id, args, run):
    audio_loader, text_loader = dataloaders
    
    total_steps = 0
    audio_steps = 0
    text_steps = 0
    accum_step = 0
    epoch = 0
    
    num_batch = len(audio_loader) # + len(text_loader)
    print(f"Starting training with {args.steps} steps, which corresponds to {args.epoch} epochs")
    
    for epoch in range(args.epoch):
        step_bar = tqdm(range(args.steps), desc=f"Epoch: {epoch + 1}/{args.epoch}", unit="step")
        audio_iter = iter(audio_loader)
        # text_iter = iter(text_loader)
        text_iter = None
        while total_steps < args.steps:
            # use_audio = total_steps % 2 == 0
            use_audio = True  # TODO change      
            try:
                batch, batch_type = get_next_batch(use_audio, audio_iter, text_iter)
                if use_audio: audio_steps += 1
                else: text_steps += 1
            except StopIteration:
                print("WARNING: Data exhausted before reaching the target steps, starting a new epoch.")
                break
            
            batch_tensor = batch['tensor'].to(args.device)
            batch_tensor = batch_tensor.long()

            if accum_step == 0:
                optimizers['projector'].zero_grad()
                optimizers['finetune'].zero_grad()

            # We don't need to shift input as we use Forced Teaching
            target_seq = batch_tensor[:, :, 1:]

            text_logits, audio_logits = model.forward(batch_tensor)

            if torch.isnan(text_logits).any() or (audio_logits is not None and torch.isnan(audio_logits).any()):
                print("WARNING: NaN detected in model outputs!")
                continue
            
            total_loss = calculate_losses(text_logits, audio_logits, target_seq, batch_type, pad_id, args)

            if torch.isnan(total_loss).any():
                print("WARNING: NaN detected in loss calculation!")
                continue
            
            # Normalize loss by gradient accumulation steps to maintain scale
            total_loss = total_loss / args.accumulation
            if run:
                run["train/loss"].log(total_loss.item() * args.accumulation)

            total_loss.backward()

            mem_used = torch.cuda.max_memory_allocated() / 1024**2

            if check_grad_nan(model):
                accum_step = 0
                continue
            
            accum_step += 1        
            if accum_step == args.accumulation:
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizers['projector'].step()
                optimizers['finetune'].step()

                if check_param_nan(model):
                    print("WARNING: Ending training due to NaN in parameters.")
                    step_bar.close()
                    break
                
                schedulers['projector'].step()
                schedulers['finetune'].step()

                step_bar.set_postfix({
                    "Loss": f"{total_loss.item() * args.accumulation:.4f}",
                #     "Batch": batch_type,
                    "Max Mem(MB)": f"{mem_used:.2f}",
                })
                step_bar.update(1)
                total_steps += 1
                accum_step = 0
    
    print(f"Training completed. Total epochs: {epoch}, Total steps: {total_steps}, Audio steps: {audio_steps}, Text steps: {text_steps}")
    save_checkpoint(model, optimizers, schedulers, total_steps, args)

def main():
    args = get_args()
    if args.steps == 0 and args.epoch == 0:
        raise ValueError("Either --steps or --epoch must be specified.")
    if args.steps > 0 and args.epoch > 0:
        raise ValueError("Only one of --steps or --epoch can be specified.")

    if args.neptune:
        neptune_token = os.getenv("NEPTUNE_API_TOKEN")
        run = neptune.init_run(
            project="slavivo/Thesis",
            api_token=neptune_token,
        )
        run["parameters"] = args
    else:
        run = None

    seed_all(args.seed)
    dataloaders = create_data_loaders(args)
    steps_per_epoch = len(dataloaders[0]) # TODO maybe add + len(dataloaders[1]) 
    if args.steps > 0:
        args.epoch = args.steps // steps_per_epoch + 1 
    else:
        args.steps = args.epoch * steps_per_epoch # TODO maybe add len(dataloaders[1]) as well    
    model, tokenizer = initialize_model(args.device, args.moshi_weight, args.qwen_weight)
    optimizers, schedulers = create_optimizers_and_schedulers(model, args, steps_per_epoch * args.warmup)
    os.makedirs(args.save_dir, exist_ok=True)

    train(model, dataloaders, optimizers, schedulers, tokenizer.pad_token_id, args, run)
    if run:
        run.stop()


if __name__ == "__main__":
    main()