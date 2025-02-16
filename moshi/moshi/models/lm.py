from dataclasses import dataclass
from functools import partial
import typing as tp
import torch
from torch import nn

from ..utils.sampling import sample_token
from ..utils.compile import CUDAGraphed
from ..modules.streaming import StreamingContainer, StreamingModule
from ..modules.transformer import (
    Transformer,
    StreamingTransformer,
    create_norm_fn,
)

class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
    """

    def __init__(self, *args, norm: bool = False, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        return y


class BaseLMModel(nn.Module):
    """Base class for both streaming and non-streaming LM models."""
    
    def __init__(
        self,
        delays: tp.List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: tp.Optional[int] = None,
        context: tp.Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert len(delays) == self.num_codebooks, "unexpected number of delays"
        self.delays = delays
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_scale = hidden_scale
        self.norm = norm
        self.bias_proj = bias_proj
        self.depformer_dim = depformer_dim
        self.depformer_dim_feedforward = depformer_dim_feedforward
        self.depformer_multi_linear = depformer_multi_linear
        self.depformer_weights_per_step = depformer_weights_per_step
        self.depformer_pos_emb = depformer_pos_emb
        self.existing_text_padding_id = existing_text_padding_id
        self.context = context
        self.device_type = device
        self.dtype = dtype
        self.out_norm = create_norm_fn(norm, dim)
        
        # Initialize embeddings
        self._init_embeddings(norm_emb, device, dtype)

        # Initialize args
        self._init_args(kwargs, depformer_pos_emb, depformer_weights_per_step)
        
        # Initialize depformer stuff
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(dep_q)]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )
        if self.depformer_dim_feedforward is None:
            self.depformer_dim_feedforward = int(hidden_scale * depformer_dim)

        self.linears = nn.ModuleList(
            [nn.Linear(depformer_dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )
        
        # Initialize transformers and projections
        self._init_transformers()

    def _init_embeddings(self, norm_emb, device, dtype):
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        self.emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, self.dim) for _ in range(self.n_q)]
        )
        extra_text = self.existing_text_padding_id is None
        self.text_emb = EmbeddingFactory(self.text_card + 1, self.dim)
        self.text_linear = nn.Linear(self.dim, self.text_card + extra_text, bias=self.bias_proj)
        
        self.depformer_emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, self.depformer_dim) for _ in range(self.dep_q - 1)]
        )
        self.depformer_text_emb = EmbeddingFactory(self.text_card + 1, self.depformer_dim)

    def _init_args(self, kwargs, depformer_pos_emb, depformer_weights_per_step):
        kwargs["context"] = self.context # TODO check this

        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        dep_kwargs = main_kwargs.copy()
        dep_kwargs.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        dep_kwargs["positional_embedding"] = depformer_pos_emb
        dep_kwargs["context"] = None
        if depformer_weights_per_step:
            dep_kwargs["weights_per_step"] = self.dep_q

        self.main_kwargs = main_kwargs
        self.dep_kwargs = dep_kwargs
    
    def _init_transformers(self):
        raise NotImplementedError("Implement in subclass")

    @property
    def initial_token_id(self) -> int:
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        return self.text_card if self.existing_text_padding_id is None else self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        return 0

    @property
    def zero_token_id(self) -> int:
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        return -2
    
    @property
    def device(self):
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        device = next(iter(self.parameters())).device
        zero = torch.full([1, 1, 1], self.zero_token_id, device=device, dtype=torch.long)
        special = torch.full_like(zero, self.initial_token_id)
        text_special = torch.full_like(zero, self.text_initial_token_id)
        
        audio_token = special.expand(-1, self.num_audio_codebooks, -1)
        return torch.cat([text_special, audio_token], dim=1)

    def forward_text(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, S = sequence.shape
        assert K == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {K}"
        
        input_ = None # Shape [B, S, dim]
        # Audio embeddings
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](sequence[:, cb_index + self.audio_offset])
            input_ = audio_emb if input_ is None else input_ + audio_emb
            
        # Text embeddings
        text_emb = self.text_emb(sequence[:, 0])
        input_ = text_emb if input_ is None else input_ + text_emb
        
        # Forward through transformer
        transformer_out = self.transformer(input_)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
            
        text_logits = self.text_linear(transformer_out)[:, None]
        return transformer_out, text_logits

    def forward_depformer(self, depformer_cb_index: int, sequence: torch.Tensor,
                         transformer_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in subclass")

class StreamingLMModel(BaseLMModel, StreamingContainer):
    def _init_transformers(self):
        self.transformer = StreamingTransformer(
            d_model=self.dim,
            num_heads=self.num_heads,
            dim_feedforward=int(self.hidden_scale * self.dim),
            norm=self.norm,
            device=self.device_type,
            dtype=self.dtype,
            **self.main_kwargs
        )
        
        self.depformer = StreamingTransformer(
            d_model=self.depformer_dim,
            dim_feedforward=self.depformer_dim_feedforward,
            norm=self.norm,
            device=self.device_type,
            dtype=self.dtype,
            **self.dep_kwargs
        )
        self.depformer.set_streaming_detached(True)

    def forward_depformer(self, depformer_cb_index: int, sequence: torch.Tensor,
                         transformer_out: torch.Tensor) -> torch.Tensor:
        B, K, S = sequence.shape
        assert K == 1 and S == 1, f"Must pass codebooks one by one in streaming mode, got {K} codebooks and {S} steps"
        assert transformer_out.shape[1] == 1, "Transformer out should be for a single step"
        # Project and add embeddings
        depformer_input = self.depformer_in[depformer_cb_index if self.depformer_multi_linear else 0](transformer_out)
        last_token_input = (
            self.depformer_text_emb(sequence[:, 0]) if depformer_cb_index == 0
            else self.depformer_emb[depformer_cb_index - 1](sequence[:, 0])
        )
        depformer_input = depformer_input + last_token_input
        
        # Forward through depformer
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)[:, None]
        
        assert logits.dim() == 4  # [B, Ka, S, card]
        return logits

class LMModel(BaseLMModel):
    def _init_transformers(self):
        self.transformer = Transformer(
            d_model=self.dim,
            num_heads=self.num_heads,
            dim_feedforward=int(self.hidden_scale * self.dim),
            norm=self.norm,
            device=self.device_type,
            dtype=self.dtype,
            **self.main_kwargs
        )
        
        self.depformer = Transformer(
            d_model=self.depformer_dim,
            dim_feedforward=self.depformer_dim_feedforward,
            norm=self.norm,
            device=self.device_type,
            dtype=self.dtype,
            **self.dep_kwargs
        )

    def forward_depformer(self, depformer_cb_index: int, sequence: torch.Tensor,
                         transformer_out: torch.Tensor) -> torch.Tensor:
        B, K, S = sequence.shape
        assert K == 1, "Must pass codebooks one by one"
        
        # Project and add embeddings
        depformer_input = self.depformer_in[depformer_cb_index if self.depformer_multi_linear else 0](transformer_out)
        last_token_input = (
            self.depformer_text_emb(sequence[:, 0]) if depformer_cb_index == 0
            else self.depformer_emb[depformer_cb_index - 1](sequence[:, 0])
        )
        depformer_input = depformer_input + last_token_input
        
        # Forward through depformer
        depformer_input = depformer_input.view(B*S, 1, -1) # [B, S, dim] -> [B*S, 1, dim]
        dep_output = self.depformer(depformer_input, depformer_cb_index)
        dep_output = dep_output.view(B, S, -1) # [B*S, 1, dim] -> [B, S, dim]
        logits = self.linears[depformer_cb_index](dep_output)[:, None]
        
        assert logits.dim() == 4  # [B, Ka, S, card]
        return logits

# Similar refactoring can be applied to LMGen and StreamingLMGen classes
class BaseLMGen:
    """Base class for both streaming and non-streaming LM generators."""
    def __init__(
        self,
        lm_model: BaseLMModel,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
    ):
        assert not lm_model.training
        self.lm_model = lm_model
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.check = check
        self.max_delay = max(lm_model.delays)
        self.delays_cuda = torch.tensor(lm_model.delays, device=lm_model.device, dtype=torch.long)

    def _sample_token(self, logits: torch.Tensor, is_text: bool = False) -> torch.Tensor:
        """Helper method to sample tokens based on logits."""
        return sample_token(
            logits.float(),
            self.use_sampling,
            self.temp_text if is_text else self.temp,
            self.top_k_text if is_text else self.top_k
        )

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError("Implement in subclass")


class LMGen(BaseLMGen, nn.Module):
    """Non-streaming language model generator."""
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        BaseLMGen.__init__(self, *args, **kwargs)

    @torch.no_grad()
    def generate(self, input_tokens: torch.Tensor, verbose: bool = False) -> tp.Tuple:
        """Generate text and audio tokens from input tokens."""
        B, K, T = input_tokens.shape
        assert input_tokens.dim() == 3, "Shape should be [B, K, T]"
        assert T <= self.lm_model.context, f"Sequence length {T} exceeds context {self.lm_model.context}"
        
        # Generate text tokens
        input_ = input_tokens[:, :, :-1]  # Remove last token
        transformer_out, text_logits = self.lm_model.forward_text(input_)
        text_tokens = self._sample_token(text_logits, is_text=True)
        text_tokens = text_tokens[:, 0, :]  # B,K,T -> B,T

        # Generate audio tokens
        input_ = input_tokens[:, 0:8, :]
        # Shift text tokens to the left by one
        input_ = input_[:, :, 1:]
        audio_tokens, depth_logits = self.depformer_step(input_, transformer_out, verbose)
        return text_tokens, audio_tokens, text_logits, depth_logits

    def depformer_step(
        self,
        future_tokens: torch.Tensor,
        transformer_out: torch.Tensor,
        verbose: bool = False
    ) -> tuple[torch.Tensor, tp.Optional[list[torch.Tensor]]]:
        """Process tokens through the depformer."""
        B, K, T = future_tokens.shape
        print(f"future_tokens shape: {future_tokens.shape}")
        depformer_tokens = []
        depformer_logits = [] if verbose else None

        self.lm_model.depformer.reset_cache(B*T)

        for cb_index in range(self.lm_model.dep_q):
            input_ = future_tokens[:, cb_index, :].unsqueeze(1) # [B, dep_q, T] -> [B, 1, T]
            logits = self.lm_model.forward_depformer(cb_index, input_, transformer_out)
            
            if verbose:
                depformer_logits.append(logits)
                
            next_token = self._sample_token(logits)
            next_token = next_token[:, 0, :]  # B,K,T -> B,T
            depformer_tokens.append(next_token)

        out = torch.stack(depformer_tokens, dim=2)  # [B, T, dep_q]
        return out, depformer_logits


@dataclass
class _StreamingLMGenState:
    """State container for streaming generation."""
    cache: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_depth: CUDAGraphed
    offset: int = 0

    def reset(self):
        self.offset = 0


class StreamingLMGen(BaseLMGen, StreamingModule[_StreamingLMGenState]):
    """Streaming language model generator."""
    def __init__(self, *args, **kwargs):
        StreamingModule.__init__(self)
        BaseLMGen.__init__(self, *args, **kwargs)

    def _init_streaming_state(self, batch_size: int) -> _StreamingLMGenState:
        """Initialize streaming state."""
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, lm_model.num_codebooks, self.max_delay + 2),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )

        disable = lm_model.device != 'cuda'
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        return _StreamingLMGenState(cache, initial, graphed_main, graphed_depth)

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor) -> torch.Tensor | None:
        """Process one step of streaming generation."""
        state = self._streaming_state
        if state is None:
            raise RuntimeError("Must be used within streaming context")
            
        lm_model = self.lm_model
        B, Ki, S = input_tokens.shape
        
        assert S == 1, "Only support steps one by one"
        needed_tokens = lm_model.num_codebooks - lm_model.dep_q - 1
        assert Ki == needed_tokens, f"Expected {needed_tokens} tokens, got {Ki}"

        # Update cache with input tokens
        CT = state.cache.shape[2]
        for q_other in range(input_tokens.shape[1]):
            k = lm_model.dep_q + 1 + q_other
            delay = lm_model.delays[k]
            write_position = (state.offset + delay) % CT
            state.cache[:, k, write_position:write_position + 1] = input_tokens[:, q_other]

        # Handle initial tokens
        position = state.offset % CT
        for k, delay in enumerate(lm_model.delays):
            if state.offset <= delay:
                state.cache[:, k, position] = state.initial[:, k, 0]
                
        input_ = state.cache[:, :, position:position + 1]

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        # Generate text and audio tokens
        transformer_out, text_logits = state.graphed_main(input_)
        text_token = self._sample_token(text_logits, is_text=True)[:, 0, 0]
        audio_tokens = state.graphed_depth(text_token, transformer_out)

        # Update cache with generated tokens
        state.offset += 1
        position = state.offset % CT
        state.cache[:, 0, position] = text_token
        state.cache[:, 1:lm_model.dep_q + 1, position] = audio_tokens

        if state.offset <= self.max_delay:
            return None

        # Prepare output
        B = state.cache.shape[0]
        gen_delays_cuda = self.delays_cuda[:lm_model.dep_q + 1]
        index = (((state.offset - self.max_delay + gen_delays_cuda) % CT)
                .view(1, -1, 1)
                .expand(B, -1, 1))
        return state.cache.gather(dim=2, index=index)

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """Process tokens through the depformer in streaming mode."""
        B = text_token.shape[0]
        prev_token = text_token
        depformer_tokens = []

        assert not self.lm_model.depformer.is_streaming
        with self.lm_model.depformer.streaming(B):
            for cb_index in range(self.lm_model.dep_q):
                input_ = prev_token[:, None, None]
                logits = self.lm_model.forward_depformer(cb_index, input_, transformer_out)
                next_token = self._sample_token(logits)[:, 0, 0]
                depformer_tokens.append(next_token)
                prev_token = next_token

        out = torch.stack(depformer_tokens, dim=1)
        assert out.shape == (B, self.lm_model.dep_q)
        return out