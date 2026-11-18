// Regenerates models.js from each model's Hugging Face config.json. Run: node scripts/gen-models.js
// Gated repos (Llama, Gemma) are read from ungated mirrors with the identical config (405B: the FP8 repo, same architecture and parameter count). Needs network.
// Per entry: [repo, optional active params in B for MoE]. params_b = safetensors total from the HF API.
const fs = require("fs");
const SRC = {
  "Meta": {
    "Llama-2-7B": ["NousResearch/Llama-2-7b-hf"], "Llama-3.2-1B": ["unsloth/Llama-3.2-1B"], "Llama-3.2-3B": ["unsloth/Llama-3.2-3B"],
    "Llama-3.1-8B": ["unsloth/Llama-3.1-8B"], "Llama-3.1-70B": ["unsloth/Llama-3.1-70B"], "Llama-3.3-70B": ["unsloth/Llama-3.3-70B-Instruct"],
    "Llama-3.1-405B": ["RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8"], "CodeLlama-7B": ["codellama/CodeLlama-7b-hf"], "CodeLlama-13B": ["codellama/CodeLlama-13b-hf"],
    "CodeLlama-34B": ["codellama/CodeLlama-34b-hf"], "Llama-3.2-11B-Vision": ["unsloth/Llama-3.2-11B-Vision"], "Llama-4-Scout (MoE)": ["unsloth/Llama-4-Scout-17B-16E", 17] },
  "Mistral AI": {
    "Mistral-7B": ["mistralai/Mistral-7B-v0.1"], "Mixtral-8x7B (MoE)": ["mistralai/Mixtral-8x7B-v0.1", 12.9],
    "Mixtral-8x22B (MoE)": ["mistralai/Mixtral-8x22B-v0.1", 39.1], "Mistral-Nemo-12B": ["mistralai/Mistral-Nemo-Base-2407"],
    "Mistral-Small-24B": ["mistralai/Mistral-Small-24B-Base-2501"], "Ministral-8B": ["mistralai/Ministral-8B-Instruct-2410"] },
  "Alibaba (Qwen)": {
    "Qwen2.5-0.5B": ["Qwen/Qwen2.5-0.5B"], "Qwen2.5-1.5B": ["Qwen/Qwen2.5-1.5B"], "Qwen2.5-3B": ["Qwen/Qwen2.5-3B"], "Qwen2.5-7B": ["Qwen/Qwen2.5-7B"],
    "Qwen2.5-14B": ["Qwen/Qwen2.5-14B"], "Qwen2.5-32B": ["Qwen/Qwen2.5-32B"], "Qwen2.5-72B": ["Qwen/Qwen2.5-72B"], "Qwen3-8B": ["Qwen/Qwen3-8B"],
    "Qwen3-14B": ["Qwen/Qwen3-14B"], "Qwen3-32B": ["Qwen/Qwen3-32B"], "Qwen3-30B-A3B (MoE)": ["Qwen/Qwen3-30B-A3B", 3.3],
    "Qwen3-235B-A22B (MoE)": ["Qwen/Qwen3-235B-A22B", 22], "Qwen2.5-Coder-7B": ["Qwen/Qwen2.5-Coder-7B"], "Qwen2.5-Coder-14B": ["Qwen/Qwen2.5-Coder-14B"],
    "Qwen2.5-Coder-32B": ["Qwen/Qwen2.5-Coder-32B"], "Qwen3-Coder-30B-A3B (MoE)": ["Qwen/Qwen3-Coder-30B-A3B-Instruct", 3.3],
    "Qwen2.5-VL-7B": ["Qwen/Qwen2.5-VL-7B-Instruct"], "Qwen2.5-VL-32B": ["Qwen/Qwen2.5-VL-32B-Instruct"], "Qwen2.5-VL-72B": ["Qwen/Qwen2.5-VL-72B-Instruct"],
    "Qwen2.5-Math-7B": ["Qwen/Qwen2.5-Math-7B"] },
  "Google": {
    "Gemma-2-2B": ["unsloth/gemma-2-2b"], "Gemma-2-9B": ["unsloth/gemma-2-9b"], "Gemma-2-27B": ["unsloth/gemma-2-27b"],
    "Gemma-3-1B": ["unsloth/gemma-3-1b-pt"], "Gemma-3-4B": ["unsloth/gemma-3-4b-pt"], "Gemma-3-12B": ["unsloth/gemma-3-12b-pt"], "Gemma-3-27B": ["unsloth/gemma-3-27b-pt"] },
  "Microsoft": {
    "Phi-3-mini": ["microsoft/Phi-3-mini-4k-instruct"], "Phi-4 (14B)": ["microsoft/phi-4"], "Phi-3.5-MoE (MoE)": ["microsoft/Phi-3.5-MoE-instruct", 6.6],
    "Phi-3.5-vision": ["microsoft/Phi-3.5-vision-instruct"], "Phi-3-medium": ["microsoft/Phi-3-medium-4k-instruct"] },
  "DeepSeek": {
    "R1-Distill-Qwen-7B": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"], "R1-Distill-Llama-8B": ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"],
    "R1-Distill-Qwen-32B": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"], "R1-Distill-Llama-70B": ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B"],
    "DeepSeek-V2-Lite (MoE, MLA)": ["deepseek-ai/DeepSeek-V2-Lite", 2.4], "DeepSeek-V3 (MoE, MLA)": ["deepseek-ai/DeepSeek-V3", 37],
    "DeepSeek-Coder-6.7B": ["deepseek-ai/deepseek-coder-6.7b-base"], "DeepSeek-Coder-V2-Lite (MoE, MLA)": ["deepseek-ai/DeepSeek-Coder-V2-Lite-Base", 2.4],
    "DeepSeek-Math-7B": ["deepseek-ai/deepseek-math-7b-base"] },
  "OpenAI": { "gpt-oss-20b (MoE)": ["openai/gpt-oss-20b", 3.6], "gpt-oss-120b (MoE)": ["openai/gpt-oss-120b", 5.1] },
  "Others": {
    "OLMo-2-7B (AI2)": ["allenai/OLMo-2-1124-7B"], "Falcon3-7B (TII)": ["tiiuae/Falcon3-7B-Base"], "SmolLM2-1.7B (HF)": ["HuggingFaceTB/SmolLM2-1.7B"],
    "Yi-1.5-34B (01.AI)": ["01-ai/Yi-1.5-34B"], "InternLM2.5-7B (Shanghai AI Lab)": ["internlm/internlm2_5-7b"],
    "Kimi-K2 (Moonshot, MoE, MLA)": ["moonshotai/Kimi-K2-Instruct", 32], "StarCoder2-15B (BigCode)": ["bigcode/starcoder2-15b"],
    "SmolVLM-2B (HF)": ["HuggingFaceTB/SmolVLM-Instruct"], "Fin-R1-7B (SUFE)": ["SUFE-AIFLM-Lab/Fin-R1"],
    "OpenBioLLM-8B": ["aaditya/Llama3-OpenBioLLM-8B"], "OpenBioLLM-70B": ["aaditya/Llama3-OpenBioLLM-70B"] },
  "IBM": { "Granite-3.3-2B": ["ibm-granite/granite-3.3-2b-base"], "Granite-3.3-8B": ["ibm-granite/granite-3.3-8b-base"] },
  "NVIDIA": { "Llama-3.1-Nemotron-70B": ["nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"], "Mistral-NeMo-Minitron-8B": ["nvidia/Mistral-NeMo-Minitron-8B-Base"] },
  "Zhipu AI": { "GLM-4.5-Air (MoE)": ["zai-org/GLM-4.5-Air", 12] },
  "MiniMax": { "MiniMax-M2 (MoE)": ["MiniMaxAI/MiniMax-M2", 10] },
  "Baidu": { "ERNIE-4.5-21B-A3B (MoE)": ["baidu/ERNIE-4.5-21B-A3B-PT", 3] }
};
const get = async u => { const r = await fetch(u); if (!r.ok) throw new Error(r.status + " " + u); return r.json(); };
(async () => {
  const out = {}, sources = {}, failed = [];
  for (const [creator, models] of Object.entries(SRC)) {
    out[creator] = {};
    for (const [name, [repo, active]] of Object.entries(models)) {
      try {
        const [cfg0, api] = await Promise.all([get(`https://huggingface.co/${repo}/raw/main/config.json`), get(`https://huggingface.co/api/models/${repo}`)]);
        const c = { ...(cfg0.text_config || {}), ...cfg0 }; // multimodal configs nest the language model
        const t = cfg0.text_config || cfg0, layers = t.num_hidden_layers ?? c.num_hidden_layers;
        const mla = t.kv_lora_rank != null;
        const kv = mla ? 1 : (t.num_key_value_heads ?? t.num_attention_heads ?? c.num_attention_heads);
        const hd = mla ? (t.kv_lora_rank + t.qk_rope_head_dim) / 2 : (t.head_dim ?? t.hidden_size / t.num_attention_heads);
        const total = api.safetensors?.total; if (!total) throw new Error("no safetensors total");
        out[creator][name] = [+(total / 1e9).toFixed(2), layers, kv, hd, ...(active ? [active] : [])];
        sources[name] = `https://huggingface.co/${repo}/blob/main/config.json`;
      } catch (e) { failed.push(`${name} (${repo}): ${e.message}`); }
    }
  }
  const j = o => JSON.stringify(o).replace(/\],"/g, "],\n  \"").replace(/\{"/, "{\n  \"");
  fs.writeFileSync(__dirname + "/../models.js", `// GENERATED by scripts/gen-models.js from each model's Hugging Face config.json. Do not edit by hand.
// [params_b, layers, kv_heads, head_dim, active_params_b if MoE]. params_b = safetensors total; MoE active params are from model cards.
// MLA models (DeepSeek, Kimi): kv_heads=1, head_dim=(kv_lora_rank+rope_dim)/2 is an EFFECTIVE value so the KV formula matches their compressed cache.
// Gated repos (Llama, Gemma) are read from ungated mirrors with identical configs. Caveat: sliding-window/chunked-attention models (Gemma, Llama 4) are overestimated at long context. Gated repos with no ungated mirror (Cohere Command R, Mistral Large) are not listed.
const MODELS_BY_CREATOR = ${JSON.stringify(out, null, 1).replace(/\n\s+(?=[\d\]\[])/g, " ")};
const MODEL_SOURCES = ${JSON.stringify(sources, null, 1)};
`);
  console.log("written; failed:", failed.length ? "\n" + failed.join("\n") : "none");
})();
