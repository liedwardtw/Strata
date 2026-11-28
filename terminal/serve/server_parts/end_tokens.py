"""Moved unchanged out of serve/server.py (file split, cleanup step 5): the end tokens part. `serve.server` imports these names back,
so every caller and test still reaches them as `serve.server.<name>`."""
from __future__ import annotations


from serve.server_parts.base import IM_END


# The end token a tokenizer names, under the key this pack writes it with.  A tokenizer.json made by llama.cpp's
# conversion keeps GGUF's own key ("tokenizer.ggml.eos_token_id"); a hand-made pack may shorten it to "eos".
EOS_KEYS = ("eos", "tokenizer.ggml.eos_token_id", "eot", "tokenizer.ggml.eot_token_id")


def end_token_ids(tokenizer) -> set[int]:
    """The ids that end a reply for a tokenizer whose family was not named.

    Qwen's <|im_end|> and <|endoftext|> end a reply only where each is ONE token.  A tokenizer without them (one that
    splits "<|im_end|>" into '<', '|', 'im', '_end', '|', '>') would make every piece of them a stop id: the reply ended in front
    of the first "|", ">", "end", "of", "text" or "im" it wrote, so "Nerazim" came out as "Neraz" (#4576, and a table's first
    "|" in #4137).  Those ids are left out, and the end token the tokenizer names (special_ids "eos") ends the reply instead.

    The named end token is always a stop id, not only when neither marker is one token: a model whose own end-of-turn
    token is neither <|im_end|> nor <|endoftext|> (Inkling 2 ends a turn with <|content_model_end_sampling|>, id 200006)
    stops there in the engine, and a server that does not know that id keeps reading until max_new and reports the reply
    as cut short at the token cap.  Both are the end of the reply, so both are stop ids."""
    ids: set[int] = set()
    for marker in (IM_END, "<|endoftext|>"):
        got = tokenizer.encode(marker, parse_special=True)
        if len(got) == 1:
            ids.add(int(got[0]))
    named = getattr(tokenizer, "special_ids", None) or {}
    for key in EOS_KEYS:
        eos = named.get(key)
        if eos is not None:
            ids.add(int(eos))
    return ids
