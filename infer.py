"""Chat with a Thonk checkpoint.

Two modes, picked by the checkpoint:

- Base model (train.py): nothing is added to what you type. No template, no
  system prompt — your text is tokenized verbatim and the model simply
  continues it. Prompt accordingly ("The capital of France is"). An empty
  line starts generation from a document boundary.

- Chat model (chat_sft.py, saved with "chat": True): a real multi-turn
  conversation in the ChatML template it was fine-tuned on, with an optional
  system prompt. Each reply ends at <|im_end|>. An empty line clears the
  history.

Run: uv run infer.py
"""

import glob
import os

import torch

from components.dataset.chat import IM_END, chat_enc, prompt_ids
from components.model import Thonk

# ---------------- settings ----------------
checkpoint_dir = "checkpoints"  # "checkpoints_chat" for the fine-tuned model
checkpoint_path = ""  # a specific file, or empty for the newest in checkpoint_dir
max_new_tokens = 256
temperature = 0.8
top_k = 50  # 0 disables top-k filtering
system_prompt = ""  # chat models only, e.g. "Your persona:\nI love hiking."
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------

# GPT-2 BPE plus the two ChatML tokens; a base model just never sees them.
enc = chat_enc


def load_model() -> tuple[Thonk, bool]:
    path = checkpoint_path or max(
        glob.glob(os.path.join(checkpoint_dir, "step_*.pt")), default=None
    )
    if path is None:
        raise SystemExit(f"no checkpoints in {checkpoint_dir}/ — run train.py first")
    # weights_only=False: the checkpoint pickles our ThonkConfig dataclass,
    # and it's our own file from train.py.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"]
    # A checkpoint saved from a torch.compile'd model prefixes every key.
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model = Thonk(ckpt["config"]).to(device)
    model.load_state_dict(state)
    model.eval()
    is_chat = ckpt.get("chat", False)
    print(f"loaded {path} (step {ckpt['step']}, {'chat' if is_chat else 'base'} model)")
    return model, is_chat


@torch.no_grad()
def generate(model: Thonk, ids: torch.Tensor, stop_id: int) -> str:
    """Sample up to max_new_tokens continuing `ids`, streaming text to stdout.

    Stops at stop_id (EOT for a base model, <|im_end|> for a chat model) and
    returns the generated text.

    No KV cache yet — every step re-runs the full prefix, which is fine at
    interactive lengths. Caching MLA's compressed latent (the whole point of
    MLA at inference) is the upgrade to make here later.
    """
    generated = []
    printed = 0
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(ids)
        logits = logits[0, -1].float()

        if temperature == 0:
            next_id = logits.argmax()
        else:
            logits = logits / temperature
            if top_k:
                kth_best = torch.topk(logits, top_k).values[-1]
                logits[logits < kth_best] = -float("inf")
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)[0]

        if next_id.item() in (stop_id, enc.eot_token):
            break
        generated.append(next_id.item())
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)

        # Stream by re-decoding everything and printing the new suffix; a
        # token can end mid-way through a multi-byte character, so hold off
        # while the decode ends in the replacement char.
        text = enc.decode(generated)
        if not text.endswith("�"):
            print(text[printed:], end="", flush=True)
            printed = len(text)
    print()
    return enc.decode(generated)


def chat(model: Thonk):
    print("chat: empty line clears the history. Ctrl-D to exit.")
    system = [{"role": "system", "content": system_prompt}] if system_prompt else []
    history = []
    while True:
        try:
            message = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            history = []
            print("(history cleared)")
            continue
        history.append({"role": "user", "content": message})
        ids = torch.tensor(prompt_ids(system + history), device=device).unsqueeze(0)
        reply = generate(model, ids, stop_id=IM_END)
        history.append({"role": "assistant", "content": reply})


def main():
    model, is_chat = load_model()
    if is_chat:
        chat(model)
        return
    print("pretrain chat: your text is the prompt, verbatim. Ctrl-D to exit.")
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        # encode_ordinary: no special tokens added — the prompt is exactly
        # what you typed. Empty prompt: generate a fresh "document" from EOT,
        # which doubles as the document separator in training data.
        prompt_ids = enc.encode_ordinary(prompt) or [enc.eot_token]
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        generate(model, ids, stop_id=enc.eot_token)


if __name__ == "__main__":
    main()
