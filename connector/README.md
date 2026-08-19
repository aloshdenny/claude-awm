# Invisible Ink connector

A local MCP server that gives Claude one real capability: score text against
this repo's research SynthID watermark key and return a verdict, z-score,
and a per-token g-value heatmap. It reuses the exact detection math from
[src/synthid_robustness.py](../src/synthid_robustness.py), same key seeding,
same canonical sampling table, same mean-g/z-score, so the numbers it
returns are faithful to the study.

## What it does and does not do

**Does:** detect whether text was watermarked with *this connector's own
research key*, and show you the per-token signal that drives that call.

**Does not:** tell you whether text really came from Claude, ChatGPT,
Gemini, or any production system. Detection requires the exact key used at
generation time, and nobody outside the company that issued it has that
key. Paste in real AI output and it will correctly read as "not detected,"
because it wasn't generated with this key. That is the same soundness
property the whole study relies on, not a limitation of this tool
specifically.

There is no companion tool that hands back a modified or "cleaned" version
of your text. This connector is detection-only, on purpose. The attack
transform lives separately in [site/](../site/), where it never claims a
detection score on arbitrary pasted text either.

## Setup

```bash
cd connector
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

The first call downloads the tokenizer for the default model
(`Qwen/Qwen3.5-0.8B`, tokenizer files only, no model weights) and caches it
locally.

## Add it to Claude Code

```bash
claude mcp add invisible-ink -- python /absolute/path/to/claude-awm/connector/server.py
```

## Add it to Claude Desktop

In your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "invisible-ink": {
      "command": "python",
      "args": ["/absolute/path/to/claude-awm/connector/server.py"]
    }
  }
}
```

Restart Claude Desktop, then ask it to analyze some text. Try pasting one of
the watermarked samples from [site/demo_samples.json](../site/demo_samples.json)
to see a real positive.

## Tools

- `analyze_watermark(text, model="Qwen/Qwen3.5-0.8B")` — verdict, z-score,
  mean g, and a text heatmap for the given text.
- `explain_finding()` — a short summary of the study's headline result,
  static, no computation.
