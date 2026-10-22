"""Worker settings for a smoke test from the model's vLLM recipe, when one exists.

vLLM's recipes (rendered at recipes.vllm.ai) are YAML files at
models/<org>/<repo>.yaml in vllm-project/recipes; launch-day recipes can also
live under recipes/ in vllm-project/vllm. The schema is SCHEMA.md in
vllm-project/recipes.

Only settings that decide whether the model starts on one GPU are applied:
backend, dtype and loading choices, plus the recipe's env vars. Parallelism,
context length, batch limits, parsers and opt-in features belong to the real
deploy, so they are listed as skipped instead. --trust-remote-code is never
applied from a recipe; the smoke ladder's remote-code rules decide that.

Precedence, lowest first: model.base_args/base_env, the default variant's
extra_args/extra_env, the hardware override for the GPU generation (or the
brand-wide "nvidia" block when there is no generation block), the default
variant's hardware override, and for exact GPU ids the variant's override for
that id. strategy_overrides.single_node_tp.hardware_overrides.<generation>, when
present, replaces the top-level block for that generation (Step-5-Preview uses
it to drop a MoE backend that needs expert parallelism).
"""

import urllib.error
import urllib.request

SOURCES = (
    "https://raw.githubusercontent.com/vllm-project/recipes/main/models/{model}.yaml",
    "https://raw.githubusercontent.com/vllm-project/vllm/main/recipes/models/{model}.yaml",
)
UA = "day0-ready/0.1 (+https://github.com/ashwinsreedhar28/day0-ready)"

# Flags that change whether or how the model starts on one GPU -> worker env.
APPLY = {"--attention-backend", "--moe-backend", "--kv-cache-dtype", "--dtype", "--quantization",
         "--gpu-memory-utilization", "--block-size", "--enforce-eager", "--hf-overrides", "--tokenizer-mode",
         "--config-format", "--load-format"}
BOOL_FLAGS = {"--enforce-eager"}

# Runpod GPU type -> (generation, recipe hardware profile id or None).
GPU_PROFILES = (
    ("H100", "hopper", "h100"), ("H200", "hopper", "h200"), ("B200", "blackwell", "b200"),
    ("L40S", "ada", "l40s"), ("4090", "ada", None), ("L40", "ada", None), ("RTX 6000 Ada", "ada", None),
    ("A100", "ampere", None), ("A6000", "ampere", None), ("A5000", "ampere", None), ("3090", "ampere", None),
)


def fetch(model, sources=SOURCES, timeout=20):
    """(recipe dict, source URL), or (None, None) when no source has one."""
    import yaml  # PyYAML; only needed when a recipe is actually read

    for src in sources:
        url = src.format(model=model)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return yaml.safe_load(r.read()), url
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    return None, None


def hardware(gpu_types):
    """(generation, profile id) shared by every GPU type in the list; None where they differ or are unknown."""
    found = []
    for t in gpu_types:
        hit = next(((gen, prof) for key, gen, prof in GPU_PROFILES if key in t), (None, None))
        found.append(hit)
    gens, profs = {g for g, _ in found}, {p for _, p in found}
    return (gens.pop() if len(gens) == 1 else None), (profs.pop() if len(profs) == 1 else None)


def parse_args(args):
    """['--x', 'v', '--flag', '--y=w'] -> [('--x', 'v'), ('--flag', None), ('--y', 'w')]."""
    out, i, args = [], 0, [str(a) for a in args or []]
    while i < len(args):
        a = args[i]
        if not a.startswith("--"):
            i += 1  # stray value; nothing to attach it to
            continue
        if "=" in a:
            flag, value = a.split("=", 1)
            out.append((flag, value))
            i += 1
        elif i + 1 < len(args) and not args[i + 1].startswith("--"):
            out.append((a, args[i + 1]))
            i += 2
        else:
            out.append((a, None))
            i += 1
    return out


def _block(overrides, generation):
    overrides = overrides or {}
    if generation and generation in overrides:
        return overrides[generation] or {}
    return overrides.get("nvidia") or {}


def settings(recipe, gpu_types):
    """Worker env from a recipe for one GPU class: {'env': {...}, 'skipped': [...], 'notes': [...]}."""
    generation, profile = hardware(gpu_types)
    model = recipe.get("model") or {}
    variant = (recipe.get("variants") or {}).get("default") or {}
    top = dict(recipe.get("hardware_overrides") or {})
    single = ((recipe.get("strategy_overrides") or {}).get("single_node_tp") or {}).get("hardware_overrides") or {}
    if generation in single:
        top[generation] = single[generation]
    layers = [(model.get("base_args"), model.get("base_env")),
              (variant.get("extra_args"), variant.get("extra_env"))]
    blk = _block(top, generation)
    layers.append((blk.get("extra_args"), blk.get("extra_env")))
    vblocks = variant.get("hardware_overrides") or {}
    vblk = _block(vblocks, generation)
    layers.append((vblk.get("extra_args"), vblk.get("extra_env")))
    if profile and profile in vblocks:
        layers.append(((vblocks[profile] or {}).get("extra_args"), (vblocks[profile] or {}).get("extra_env")))

    env, skipped, notes = {}, [], []
    for args, extra_env in layers:
        for flag, value in parse_args(args):
            if flag == "--trust-remote-code":
                notes.append("recipe passes --trust-remote-code (left to the ladder's remote-code rules)")
            elif flag in APPLY:
                name = flag[2:].replace("-", "_").upper()
                env[name] = "true" if flag in BOOL_FLAGS or value is None else value
            else:
                skipped.append(flag if value is None else f"{flag} {value}")
        for k, v in (extra_env or {}).items():
            env[str(k)] = str(v)
    if model.get("min_vllm_version"):
        notes.append(f"min vLLM {model['min_vllm_version']}")
    hw = (recipe.get("meta") or {}).get("hardware") or {}
    if profile and hw.get(profile) == "unsupported":
        notes.append(f"recipe marks {profile} unsupported")
    return {"env": env, "skipped": sorted(set(skipped)), "notes": sorted(set(notes)), "generation": generation}
