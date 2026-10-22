"""Run `vllm serve` on a local GGUF with one startup check skipped.

vllm 0.32.0rc1.dev348 (vllm/vllm-openai:flint) calls
maybe_override_with_speculators() on the model path, which opens a local .gguf
file as if it were config.json and fails ("... is not a valid JSON file").
This skips that check for .gguf paths only. Run as:
    python3 vllm_gguf.py serve /root/models/flint-nvfp4.gguf ...
"""
import sys

import vllm.engine.arg_utils as a
import vllm.transformers_utils.config as c

orig = c.maybe_override_with_speculators


def skip_for_gguf(model, tokenizer, *args, **kw):
    if str(model).endswith(".gguf"):
        return model, tokenizer, kw.get("vllm_speculative_config")
    return orig(model, tokenizer, *args, **kw)


c.maybe_override_with_speculators = skip_for_gguf
a.maybe_override_with_speculators = skip_for_gguf

if __name__ == "__main__":  # vLLM's engine process re-imports this file
    from vllm.entrypoints.cli.main import main

    sys.argv = ["vllm"] + sys.argv[1:]
    sys.exit(main())
