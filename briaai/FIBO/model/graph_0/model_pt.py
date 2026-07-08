import torch

MODEL_ID = "briaai/FIBO"
DATA_FORMAT = torch.bfloat16
BATCH_SIZE = 2
GUIDANCE_SCALE = 5.0
NUM_INFERENCE_STEPS = 50
SEED = 42

BRINGUP_PROMPT = (
    '{"subject":"a hyper-detailed, ultra-fluffy owl in moonlit trees",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)


class _ShortCircuit(Exception):
    """Raised inside the patched transformer to abort the pipeline after step 0."""


class FiboTransformerWrapper(torch.nn.Module):
    """Replay the captured (args, kwargs) call shape against the FIBO DiT.

    The benchmark calls model(*inputs) positionally; this splits the flat
    positional tuple back into the transformer's args + kwargs.
    """

    def __init__(self, transformer, capture):
        super().__init__()
        self.transformer = transformer
        self._num_args = len(capture["args"])
        self._kwarg_keys = tuple(capture["kwargs"].keys())

    def forward(self, *inputs):
        expected = self._num_args + len(self._kwarg_keys)
        if len(inputs) != expected:
            raise ValueError(
                f"FiboTransformerWrapper expected {expected} positional inputs "
                f"(got {len(inputs)})."
            )
        args = inputs[: self._num_args]
        kwargs = dict(zip(self._kwarg_keys, inputs[self._num_args :]))
        out = self.transformer(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            return out[0]
        if hasattr(out, "sample"):
            return out.sample
        return out


_PIPE = None
_CAPTURE = None


def _load_pipe():
    try:
        from diffusers import BriaFiboPipeline
    except ImportError:
        from diffusers import DiffusionPipeline as BriaFiboPipeline

    pipe = BriaFiboPipeline.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)
    pipe.to("cpu")
    for attr in ("text_encoder", "transformer", "vae"):
        module = getattr(pipe, attr, None)
        if module is None:
            continue
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False
    return pipe


def _capture_transformer_inputs(pipe):
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("FIBO pipeline does not expose a .transformer attribute.")

    capture = {}
    original_forward = pipe.transformer.forward

    def patched_forward(*args, **kwargs):
        capture["args"] = args
        capture["kwargs"] = kwargs
        out = original_forward(*args, **kwargs)
        capture["output"] = out
        raise _ShortCircuit()

    pipe.transformer.forward = patched_forward
    try:
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        with torch.no_grad():
            try:
                pipe(
                    prompt=BRINGUP_PROMPT,
                    negative_prompt=None,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                    output_type="latent",
                )
            except _ShortCircuit:
                pass
    finally:
        pipe.transformer.forward = original_forward

    if "args" not in capture:
        raise RuntimeError("FIBO pipeline never invoked the patched transformer.")
    return capture


def _ensure_pipe_and_capture():
    global _PIPE, _CAPTURE
    if _CAPTURE is not None:
        return _PIPE, _CAPTURE
    pipe = _load_pipe()
    capture = _capture_transformer_inputs(pipe)
    _PIPE, _CAPTURE = pipe, capture
    return _PIPE, _CAPTURE


def load_pytorch_model():
    pipe, capture = _ensure_pipe_and_capture()
    pipe.transformer = pipe.transformer.to(DATA_FORMAT)
    return FiboTransformerWrapper(pipe.transformer, capture).eval()


def load_input():
    """Replay positional_inputs_from_capture + load_inputs' float-only cast.

    Returns a positional tuple:
      (hidden_states, timestep, encoder_hidden_states, text_encoder_layers,
       joint_attention_kwargs, return_dict, txt_ids, img_ids)
    Non-tensor entries (joint_attention_kwargs=None, return_dict=False) and the
    text_encoder_layers list are passed through unchanged (already bf16 from the
    bf16 pipeline); only floating-point tensors are cast to DATA_FORMAT.
    """
    _, capture = _ensure_pipe_and_capture()
    flat = list(capture["args"])
    for value in capture["kwargs"].values():
        flat.append(value)

    cast = []
    for value in flat:
        if torch.is_tensor(value) and value.is_floating_point():
            cast.append(value.to(DATA_FORMAT))
        else:
            cast.append(value)
    return tuple(cast)


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input()

    with torch.no_grad():
        output = model(*inputs)

    return output
