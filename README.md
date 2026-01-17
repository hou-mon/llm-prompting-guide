# The Complete Guide to LLM Prompting
## From "Let's Think Step by Step" to Recursive Language Models

**A practical guide for everyone—whether you're chatting with Claude or building the next AI agent.**

---

## Introduction

When you ask an AI a question, you're not just requesting information—you're shaping how it thinks. The way you structure a prompt can be the difference between a confused response and a breakthrough insight.

This guide covers the techniques that have transformed what's possible with language models over the past few years. Some are simple tricks you can use in any chat interface today. Others are architectural patterns for developers building sophisticated AI systems. We've organized them so you can find what's relevant to your needs.

### The Mental Model That Changes Everything

Before diving into techniques, internalize this insight from Andrej Karpathy (https://karpathy.ai):

> **Don't think of LLMs as entities—think of them as simulators.**

When exploring a topic, don't ask "What do you think about X?" There is no "you" in the way we typically mean it. The model hasn't spent years pondering X and forming opinions through lived experience.

Instead, try: **"What would be a good group of people to explore X? What would they say?"**

The LLM can channel and simulate many perspectives—an economist, a skeptic, a historian, a practitioner. When you force it via "you," it adopts a personality implied by its training data and simulates that. It's fine to do, but there's far less mystique to it than people naively attribute to "asking an AI."

This framing unlocks better prompting:
- Instead of "What's your opinion on remote work?" → "How would a CEO, a remote employee, and an organizational psychologist each view remote work?"
- Instead of "Are you sure?" → "What would a devil's advocate say about this answer?"
- Instead of "Think harder" → "Simulate an expert in [domain] analyzing this problem"

The model is a simulator. Your prompt sets the stage for what it simulates.

---

### The Big Picture: Three Eras of Prompting

**Era 1: Prompting as Instruction (2022-2023)**
We discovered that *how* you ask matters enormously. Adding "let's think step by step" or providing examples could unlock capabilities that seemed absent. The model had the knowledge—it just needed the right invitation to use it.

**Era 2: Prompting as Training Signal (2024-2025)**
The line between prompting and training blurred. Models like OpenAI's o1 and DeepSeek R1 learned to generate their own reasoning chains through reinforcement learning. Extended thinking became a learned behavior, not just a prompted one.

**Era 3: Prompting as Environment (2025+)**
The latest frontier: models that treat their own context as an external environment to be programmatically explored. Recursive Language Models (RLMs) can examine, decompose, and call themselves over pieces of their input—handling documents 100x longer than their context window by actively managing what they attend to.

### The Unifying Insight

Across all three eras, one principle holds: **giving models more compute at inference time improves results**. Whether that's longer reasoning chains, multiple sampling paths, tree search, or recursive self-calls—more thinking time helps on hard problems.

---

## How to Use This Guide

**By Audience:**

| If you are... | Start with... |
|---------------|---------------|
| **Everyday user** wanting better ChatGPT/Claude responses | Part 1: Techniques You Can Use Today |
| **Developer** building AI-powered applications | Part 2: Reasoning Frameworks |
| **Builder** working on agents or long-context problems | Part 3: Production Agent Systems |
| **Researcher** exploring the frontier | Part 4: The Cutting Edge (2025) |
| **Reference seeker** wanting the full paper collection | Part 5: The Paper Library |

**By Progression Level:**

This guide also works as a learning progression—start at Level 1 and work through sequentially:

- **Level 1: Foundations** — Official docs, basic techniques (Part 1)
- **Level 2: Core Techniques** — CoT, Self-Consistency, Few-Shot (Part 1)
- **Level 3: Structured Reasoning** — Tree of Thoughts, Least-to-Most (Part 2)
- **Level 4: Agentic Foundations** — ReAct, Reflexion (Part 2)
- **Level 5: Production Systems** — Anthropic's patterns, agent architectures (Part 3)
- **Level 6: Programmatic Optimization** — DSPy, automatic prompting (Part 3)
- **Level 7: The Frontier** — RLMs, reasoning models, model-native agents (Part 4)

### A Note for Non-Developers

Don't let terms like "CLI," "agents," or "agentic workflows" scare you off—this isn't just for programmers. If you're a researcher, writer, analyst, farmer doing herd management, video editor logging footage, or anyone dealing with complex information, these techniques are directly applicable to your work. 

The command line is just a text box where you type instructions. An "agent" is just a model that can use tools and iterate on problems autonomously. The real unlock here is understanding how to communicate with these systems effectively. 

Chain-of-thought prompting works just as well for analyzing market trends or planning livestock rotations as it does for debugging code. ReAct-style reasoning helps any task where you need to gather information, act on it, and adjust. The frameworks in this guide will make you dangerous in whatever domain you actually care about—coding is just one application among many.

---

## Official Documentation (Start Here)

Before diving into techniques, know where the authoritative sources live:

### Anthropic's Prompt Engineering Documentation

**The gold standard for Claude-based development**

- **Main URL**: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview
- **Claude 4 Best Practices**: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices

Why it matters: Anthropic's documentation is widely regarded as the most comprehensive and practical in the industry. Covers everything from basic structure to advanced techniques.

Key pages (append to `docs.claude.com/en/docs/build-with-claude/prompt-engineering/`):
- `be-clear-and-direct` — Fundamentals of clear prompting
- `multishot-prompting` — Few-shot examples (3-5 recommended)
- `chain-of-thought` — Step-by-step reasoning
- `use-xml-tags` — Structuring prompts with XML
- `system-prompts` — Role assignment
- `long-context-tips` — Handling large documents
- `extended-thinking-tips` — For reasoning models

### Anthropic's Interactive Prompt Engineering Tutorial

**Hands-on Jupyter notebooks for learning by doing**

- **URL**: https://github.com/anthropics/prompt-eng-interactive-tutorial
- **Format**: 9 chapters with exercises + advanced appendix
- **Best for**: Developers who learn by experimentation
- **Also available as**: Google Sheets version using Claude for Sheets extension

### OpenAI Prompt Engineering Guide

**Foundation for GPT-family models**

- **URL**: https://platform.openai.com/docs/guides/prompt-engineering

Key insight: *"A reasoning model is like a senior co-worker. You can give them a goal and trust them to work out details. A GPT model is like a junior coworker—they perform best with explicit instructions."*

Model-specific guides:
- GPT-4.1: https://cookbook.openai.com/examples/gpt4-1_prompting_guide
- GPT-5: https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide
- GPT-5.2: https://cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide

### DAIR.AI Prompt Engineering Guide

**Community-maintained comprehensive reference**

- **URL**: https://www.promptingguide.ai/

Why it's essential: Aggregates techniques across all major models, continuously updated with latest papers. Coverage spans from zero-shot through advanced agent architectures.

---

## Part 1: Techniques You Can Use Today

These require nothing but changing how you write your prompts. No coding, no special tools.

### Chain-of-Thought: "Show Your Work"

**What it is**: Ask the model to reason through a problem step by step before giving an answer.

**How to use it**: Either provide an example that shows reasoning steps, or simply add "Let's think through this step by step" to your prompt.

**Example**:
```
Without CoT: "What's 17 × 24?" → Model might guess incorrectly

With CoT: "What's 17 × 24? Let's work through it step by step."
→ "17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408"
```

**When it helps**: Math problems, logic puzzles, multi-step reasoning, analyzing complex situations, debugging code.

**When it doesn't help**: Simple factual questions, creative writing (can make it feel mechanical), tasks where you want brevity.

**The research**: This was the paper that unlocked reasoning in LLMs. A 540B parameter model with 8 CoT exemplars achieved state-of-the-art on GSM8K math problems, surpassing fine-tuned GPT-3 with verifier.

📄 [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) (Wei et al., Google, 2022) — NeurIPS 2022, 8000+ citations

---

### Zero-Shot CoT: The Magic Words

**What it is**: You don't need examples. Just append **"Let's think step by step"** and reasoning often emerges automatically.

**The research**: Kojima et al. tested many phrases. "Let's think step by step" consistently worked best—better than "Let's solve this problem" or "Think carefully."

**Practical tip**: This works on models above ~100B parameters. On smaller models, it can actually hurt performance.

**Why it matters**: Democratized CoT—no need to craft exemplars.

📄 [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) (Kojima et al., 2022)

---

### Few-Shot Examples: Teaching by Demonstration

**What it is**: Show the model 2-5 examples of the input-output pattern you want before your actual question.

**Example for classification**:
```
Review: "The food was terrible" → Sentiment: Negative
Review: "Best experience ever!" → Sentiment: Positive  
Review: "It was okay, nothing special" → Sentiment: Neutral

Review: "I wouldn't recommend this to anyone" → Sentiment:
```

**Tips for better examples**:
- Use diverse examples covering edge cases
- Keep formatting consistent
- Quality matters more than quantity (3 great examples beat 10 mediocre ones)
- Put your hardest or most relevant example last

---

### Role Prompting: Set the Stage for Simulation

**What it is**: Tell the model who (or what group) it should simulate. This shapes tone, expertise level, and approach.

**Basic version**:
- "You are an experienced Python developer reviewing code for a junior engineer"
- "You are a patient teacher explaining concepts to a curious 10-year-old"

**Better version** (using the simulator framing):
- "Simulate how a senior Python developer would code-review this, noting what they'd flag and why"
- "How would a teacher known for making complex topics simple explain this to a 10-year-old?"

**Even better for exploration**:
- "I want to understand the pros and cons of microservices. Simulate a discussion between a startup CTO who loves them, a senior engineer who's been burned by them, and a pragmatic consultant. What would each say?"

This leverages the model's ability to channel many perspectives rather than forcing it into a single "opinion."

---

### Structured Output: Define the Format

**What it is**: Explicitly specify how you want the response formatted.

**Example**:
```
Analyze this business idea and respond in exactly this format:

**Strengths**: [3 bullet points]
**Weaknesses**: [3 bullet points]  
**Verdict**: [One sentence recommendation]
**Confidence**: [High/Medium/Low]
```

**Pro tip**: For complex outputs, provide a template with placeholders the model should fill in.

---

### The "Simulate a Panel" Technique

Building on Karpathy's insight, one of the most powerful techniques for exploration:

**Instead of**: "What are the risks of this investment strategy?"

**Try**: 
```
I want to evaluate this investment strategy. Simulate a panel discussion with:
- A value investor in the style of Warren Buffett
- A quantitative trader who relies on data
- A behavioral economist who studies investor psychology
- A skeptical financial journalist

What would each person say about the risks? Where would they agree and disagree?
```

This surfaces perspectives the model can simulate well, rather than asking for a single "AI opinion" that doesn't really exist.

---

## Part 2: Reasoning Frameworks (For Developers)

These techniques typically require code to implement but dramatically expand what's possible.

### Self-Consistency: Sample and Vote

**The problem**: Models can give different answers to the same question depending on random sampling. Which answer is right?

**The solution**: Generate multiple reasoning paths (say, 5-10), then take the majority answer. Correct solutions tend to converge; wrong ones scatter randomly.

**Implementation**: Set temperature > 0, sample N completions, extract the final answer from each, return the most common answer.

**Trade-off**: Multiplies your API costs by N, but can significantly boost accuracy on hard problems.

**Performance gains**: GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%)

**When to use**: Problems with verifiable answers where accuracy is critical.

📄 [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) (Wang, Wei et al., Google, 2022) — ICLR 2023, 3000+ citations

---

### Tree of Thoughts: Explore and Backtrack

**The problem**: Standard prompting commits to one reasoning path. If you make a wrong turn early, you're stuck.

**The solution**: Let the model generate multiple possible "thoughts" at each step, evaluate which are promising, and explore the tree of possibilities—with the ability to backtrack.

**Core framework**:
1. Decompose problems into "thoughts" (coherent reasoning units)
2. Generate multiple candidate thoughts at each step
3. Evaluate thoughts (value scoring or voting)
4. Search through tree (BFS, DFS, or beam search)
5. Backtrack when necessary

**Real results**: On Game of 24, GPT-4 with standard CoT solved 4% of puzzles. With Tree of Thoughts: 74%.

**Key insight**: IO, CoT, and Self-Consistency are all special cases of ToT (trees with limited depth/breadth).

**When to use**: Problems where initial choices matter, creative tasks with multiple valid approaches, puzzles requiring exploration.

**Implementation**: Requires orchestration code to manage the tree search (BFS or DFS) and evaluation prompts to score intermediate states.

**Simplified ToT prompt** (Dave Hulbert, 2023):
```
Imagine three different experts are answering this question.
All experts will write down 1 step of their thinking, then share it with the group.
Then all experts will go on to the next step, etc.
If any expert realizes they're wrong at any point then they leave.
The question is...
```

📄 [Tree of Thoughts: Deliberate Problem Solving](https://arxiv.org/abs/2305.10601) (Yao et al., Princeton & Google DeepMind, 2023) — NeurIPS 2023, 1500+ citations

---

### Least-to-Most Prompting: Decomposition for Generalization

**The problem**: Complex problems often require solving multiple interconnected subproblems.

**The solution**: 
1. Decompose problem into simpler subproblems
2. Solve subproblems sequentially, feeding solutions forward

**Advantage**: Generalizes to harder problems than those in demonstrations. Unlike CoT which can struggle when test problems are harder than examples, Least-to-Most handles complexity scaling.

**When to use**: Problems that naturally decompose, when you need to generalize beyond example difficulty.

📄 [Least-to-Most Prompting Enables Complex Reasoning](https://arxiv.org/abs/2205.10625) (Zhou, Schärli et al., Google, 2023) — ICLR 2023, 1400+ citations

---

### ReAct: Reasoning + Acting

**The pattern**: Interleave thinking and doing.

```
Thought: I need to find when the Eiffel Tower was built
Action: search("Eiffel Tower construction date")
Observation: The Eiffel Tower was completed in 1889...
Thought: Now I can answer the question
Action: finish("The Eiffel Tower was built in 1889")
```

**Why it matters**: This is the foundation of most modern AI agents. Reasoning helps the model plan what to do; actions ground it in real information from tools, databases, or APIs.

**Key insight**: Pure reasoning can hallucinate. Pure acting lacks direction. Together, they're far more capable than either alone.

**Why it's foundational**: ReAct is the architectural basis for LangChain agents, AutoGPT, and most modern agent frameworks.

**Results**: 34% absolute improvement on ALFWorld, 10% on WebShop over RL baselines.

📄 [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) (Yao et al., Princeton & Google, 2022) — ICLR 2023, 2500+ citations

---

### Reflexion: Self-Improvement Through Verbal Reinforcement

**The problem**: Agents fail, but how do they learn from failure without retraining?

**Core mechanism**:
1. Agent attempts task
2. Receives feedback (success/failure + context)
3. Generates verbal self-reflection on mistakes
4. Stores reflection in episodic memory
5. Uses memory to improve on subsequent attempts

**Key insight**: Reinforcement without weight updates—learning happens through natural language reflection stored in context.

**Results**: 97% on AlfWorld, 91% on HumanEval (from 80% baseline)

**When to use**: Iterative improvement scenarios, code generation, decision-making with feedback loops.

📄 [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., Northeastern & Princeton, 2023) — NeurIPS 2023, 900+ citations

---

## Part 3: Production Agent Systems

From research to deployment—patterns that work at scale.

### Anthropic's Building Effective Agents (2024)

**The definitive practitioner's guide**

- **URL**: https://www.anthropic.com/research/building-effective-agents
- **Authors**: Erik Schluntz and Barry Zhang

Why it's essential: Distills lessons from dozens of production deployments across industries.

**Core philosophy**: *"The most successful implementations use simple, composable patterns rather than complex frameworks."*

**Key architectural patterns**:

**1. Augmented LLM** (Basic building block)
- LLM + retrieval + tools + memory
- Focus on well-documented, specific tool interfaces

**2. Prompt Chaining**
- Sequential subtasks with validation gates
- Each step's output becomes next step's input
- Add verification between steps

**3. Routing**
- Classify input, direct to specialized handlers
- Keeps individual prompts focused

**4. Parallelization**
- Run independent subtasks simultaneously
- Aggregate results

**5. Orchestrator-Workers**
- Central LLM dynamically breaks down tasks
- Delegates to worker LLMs
- Synthesizes results

**6. Evaluator-Optimizer Loops**
- One LLM generates, another evaluates
- Iterative refinement until quality threshold

**Critical guidance**:
- Start simple, add complexity only when needed
- Understand underlying code if using frameworks
- Agentic systems trade latency/cost for performance
- Many tasks don't need agents—optimized single calls often suffice

**Code**: https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents

---

### Anthropic's Context Engineering Guide (2025)

**The evolution from prompt engineering**

- **URL**: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

**Core thesis**: *"Context engineering represents a fundamental shift—it's about what configuration of context is most likely to generate your model's desired behavior."*

**Key principles**:
- Find the smallest set of high-signal tokens that maximize desired outcomes
- System prompts: Clear, simple language at the "right altitude"
  - Avoid: Hardcoded brittle logic (too specific)
  - Avoid: Vague high-level guidance (too abstract)
- Tools: If a human can't tell which tool to use, neither can the agent
- Examples: Curate diverse, canonical examples (not edge cases)
- Token economy: Treat context as precious, finite resource

---

### Claude Agent SDK Guide (2025)

**Building production agents with Claude Code**

- **URL**: https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk

**Core design principle**: *"Give your agents a computer"*

**Agent loop**: Gather context → Take action → Verify work → Repeat

---

### Effective Harnesses for Long-Running Agents (2025)

- **URL**: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

**Problem solved**: Agents working across multiple context windows.

**Solution**:
- **Initializer agent**: Sets up environment on first run
- **Coding agent**: Makes incremental progress, leaves clear artifacts
- **Explicit testing**: Browser automation, end-to-end verification

**Key insight**: *"Compaction isn't sufficient"*—agents need explicit artifacts and handoff protocols.

---

### Multi-Agent Systems (2024-2025)

**MAR: Multi-Agent Reflexion**

- **ArXiv**: https://arxiv.org/abs/2512.20845

**Problem**: Single-agent reflection leads to "degeneration of thought"—agents spiral into unproductive patterns.

**Solution**: Multiple persona-based critics generate diverse reflections.

**Results**: 47% EM on HotPotQA, 82.7% on HumanEval

---

### DSPy: Programming, Not Prompting

**Automated prompt engineering through compilation**

- **ArXiv**: https://arxiv.org/abs/2310.03714
- **Repository**: https://github.com/stanfordnlp/dspy
- **Documentation**: https://dspy.ai

**Core concepts**:
- **Signatures**: Declarative specifications of input/output types
- **Modules**: Reusable components (ChainOfThought, ReAct, etc.)
- **Teleprompters**: Optimizers that compile prompts automatically
  - BootstrapFewShot
  - MIPRO (Multi-Stage Instruction Prompt Optimization)
  - COPRO (Cooperative Prompt Optimization)

**Why it matters**:
- Eliminates hand-crafted prompt templates
- Automatic few-shot example selection
- Cross-model portability
- GPT-3.5 with DSPy outperforms expert-prompted GPT-4 on some tasks

**Results**: 33% → 82% improvement on complex reasoning with automatic optimization.

**Related Papers**:
- DSPy Assertions: https://arxiv.org/abs/2312.13382 (Computational constraints for self-refining pipelines)
- MIPRO: Optimizing instructions for multi-stage programs

📄 [DSPy: Compiling Declarative Language Model Calls](https://arxiv.org/abs/2310.03714) (Khattab et al., Stanford NLP, 2023) — ICLR 2024, 500+ citations

---

### Production Considerations

**Code Execution with MCP**
- **URL**: https://www.anthropic.com/engineering/code-execution-with-mcp
- Key insight: LLMs writing code to interact with tools is more efficient than direct tool calls
- Benefits: 98.7% token reduction in some scenarios

**Skills Pattern**
- Reusable instruction folders for specialized tasks
- Agents can build their own toolbox of higher-level capabilities

---

## Part 4: The Cutting Edge (2025)

### Reasoning Models: Learned Thinking

**The shift**: Instead of *prompting* models to think step-by-step, we can now *train* them to do it automatically.

**DeepSeek R1** (January 2025) demonstrated that extended reasoning emerges naturally from reinforcement learning. Without any human-written reasoning examples, the model learned to:
- Break problems into steps
- Consider multiple approaches  
- Check its own work
- Backtrack when stuck

**Practical implication**: Modern reasoning models (o1, R1, Claude with extended thinking) don't need explicit CoT prompts—they reason internally. Your job shifts from "make it think" to "give it the right problem."

**The overthinking problem**: More thinking isn't always better. Research shows there's a "sweet spot"—additional reasoning beyond that point can actually decrease accuracy as the model second-guesses correct intuitions.

**Model-Specific Tips** (Claude 4.x):
- More precise instruction following than predecessors
- Request "above and beyond" behavior explicitly
- Leverage extended thinking for complex multi-step tasks
- Guide interleaved thinking for tool-use reflection:
```
After receiving tool results, carefully reflect on their quality
and determine optimal next steps before proceeding.
```

📄 [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948) (DeepSeek AI, 2025)
📄 [Does Thinking More Always Help?](https://arxiv.org/abs/2506.04210) (2025)

---

### Recursive Language Models: Context as Environment

**The problem**: Even with 128K or 1M token context windows, models struggle with very long inputs. They lose track of information, get confused, or hit hard limits.

**The RLM paradigm shift**: Instead of stuffing everything into the context window, treat the input as an **external environment** the model can programmatically explore.

**How it works**:
1. Load the full input into a variable (e.g., in a Python REPL)
2. The model never sees the whole thing in its context at once
3. It writes code to slice, search, and examine relevant portions
4. It can spawn sub-LLM calls on chunks and combine results
5. Recursively decompose until the problem is solved

**Example**: Given a 500-page legal document, the model might write:
```python
sections = split_by_headers(document)
relevant = [s for s in sections if "liability" in s.lower()]
analyses = []
for section in relevant:
    analysis = llm_query(f"Summarize liability risks: {section}")
    analyses.append(analysis)
return llm_query(f"Synthesize these findings: {analyses}")
```

**Results**: RLMs handle inputs up to 100x beyond context windows. On CodeQA, base GPT-4 scored 24%, summarization agents scored 41%, RLMs scored 66%.

**Why this matters**: The model is no longer passively receiving context—it's actively managing its own attention. This is a fundamental shift from "prompting" to "models programming themselves."

**The deeper insight**: RLMs embody the simulator framing at an architectural level. The model simulates a programmer who examines and processes the input, rather than trying to be an entity that "reads" a giant document.

📄 [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab, 2025)
🔧 Implementation: https://github.com/alexzhang13/rlm

---

### Agentic Context Engineering (ACE)

**The problem**: Long-running agents accumulate context that becomes stale, redundant, or overwhelming. "Context rot" degrades performance over time.

**ACE (Agentic Context Engineering)**: Treat context as an evolving playbook that accumulates, refines, and organizes strategies through generation, reflection, and curation.

**Key techniques**:
- Structured, incremental updates that preserve important details
- Self-curated memory that the agent maintains
- Separation of "what to remember" from "what to do now"

📄 [Agentic Context Engineering](https://arxiv.org/abs/2510.04618) (2025)

---

### The Model-Native Agent Paradigm

**The old way (pipeline-based)**: Chain together prompts, tools, and scaffolding code. The "agent" is really a workflow orchestrating an LLM.

**The new way (model-native)**: Train the model end-to-end to plan, use tools, and manage memory as intrinsic behaviors. The agent *is* the model.

**What this means for prompting**: With model-native agents, you're less likely to need elaborate prompting scaffolds. The model has internalized those patterns. Your job becomes:
1. Clearly defining the task and success criteria
2. Providing the right tools/environment
3. Setting appropriate guardrails

📄 [Beyond Pipelines: Model-Native Agentic AI](https://arxiv.org/abs/2510.16720) (2025)

---

## Quick Reference: Technique Selection

| Task Type | Recommended Technique |
|-----------|----------------------|
| Simple Q&A | Zero-shot or few-shot |
| Math/Logic | Chain-of-Thought |
| High-stakes accuracy | Self-Consistency |
| Complex decomposition | Least-to-Most |
| Exploration required | Tree of Thoughts |
| External tool use | ReAct |
| Iterative improvement | Reflexion |
| Production pipelines | DSPy |
| Multi-step workflows | Prompt Chaining |
| Long-running tasks | Agent SDK + Harnesses |

---

## Recommended Learning Path

**Week 1-2: Foundations**
- Read Anthropic's documentation thoroughly
- Complete the Interactive Tutorial
- Read Chain-of-Thought paper

**Week 3-4: Core Techniques**
- Study Self-Consistency and Zero-Shot CoT
- Implement Tree of Thoughts on a puzzle problem
- Read Least-to-Most paper

**Week 5-6: Agents**
- Critical: Read ReAct paper and implement from scratch
- Study Reflexion paper
- Read "Building Effective Agents" guide

**Week 7-8: Production**
- Work through Anthropic Cookbook examples
- Learn DSPy fundamentals
- Study Context Engineering guide
- Build a multi-step agent with tool use

---

## Part 5: The Paper Library

### Foundational Techniques

| Paper | Key Contribution | Year | Citations | Link |
|-------|------------------|------|-----------|------|
| Chain-of-Thought Prompting (Wei et al.) | Step-by-step reasoning via examples | 2022 | 8000+ | [arxiv](https://arxiv.org/abs/2201.11903) |
| Zero-Shot CoT (Kojima et al.) | "Let's think step by step" | 2022 | — | [arxiv](https://arxiv.org/abs/2205.11916) |
| Self-Consistency (Wang et al.) | Sample multiple paths, majority vote | 2022 | 3000+ | [arxiv](https://arxiv.org/abs/2203.11171) |
| Least-to-Most Prompting (Zhou et al.) | Decomposition for generalization | 2023 | 1400+ | [arxiv](https://arxiv.org/abs/2205.10625) |

### Reasoning Frameworks

| Paper | Key Contribution | Year | Citations | Link |
|-------|------------------|------|-----------|------|
| Tree of Thoughts (Yao et al.) | Explore and backtrack through reasoning | 2023 | 1500+ | [arxiv](https://arxiv.org/abs/2305.10601) |
| ReAct (Yao et al.) | Interleaved reasoning and acting | 2022 | 2500+ | [arxiv](https://arxiv.org/abs/2210.03629) |
| Reflexion (Shinn et al.) | Verbal reinforcement learning | 2023 | 900+ | [arxiv](https://arxiv.org/abs/2303.11366) |
| Plan-and-Solve (Wang et al.) | Explicit planning before execution | 2023 | — | [arxiv](https://arxiv.org/abs/2305.04091) |
| DSPy (Khattab et al.) | Compiling declarative LM programs | 2023 | 500+ | [arxiv](https://arxiv.org/abs/2310.03714) |

### Reasoning Models (2025)

| Paper | Key Contribution | Link |
|-------|------------------|------|
| DeepSeek-R1 | Reasoning emerges from RL alone | [arxiv](https://arxiv.org/abs/2501.12948) |
| R1 Thoughtology | Analysis of reasoning model behavior | [arxiv](https://arxiv.org/abs/2504.07128) |
| Does Thinking More Help? | Overthinking can hurt performance | [arxiv](https://arxiv.org/abs/2506.04210) |

### Agentic & Long-Context (2025)

| Paper | Key Contribution | Link |
|-------|------------------|------|
| Recursive Language Models | Context as programmable environment | [arxiv](https://arxiv.org/abs/2512.24601) |
| Agentic Context Engineering | Self-maintaining context playbooks | [arxiv](https://arxiv.org/abs/2510.04618) |
| Model-Native Agentic AI Survey | Pipeline vs. model-native agents | [arxiv](https://arxiv.org/abs/2510.16720) |
| A-MEM: Agentic Memory | Dynamic Zettelkasten-style memory | [arxiv](https://arxiv.org/abs/2502.12110) |
| MAR: Multi-Agent Reflexion | Multiple critics prevent degeneration | [arxiv](https://arxiv.org/abs/2512.20845) |

### Comprehensive Surveys

| Paper | Key Contribution | Link |
|-------|------------------|------|
| The Prompt Report (2024, updated 2025) | 58 techniques, 33 terms, best practices | [arxiv](https://arxiv.org/abs/2406.06608) |
| Systematic Survey of Prompt Engineering (2024) | 41 techniques by application area | [arxiv](https://arxiv.org/abs/2402.07927) |
| Demystifying Chains, Trees, Graphs (2024) | Taxonomy of reasoning topologies | [arxiv](https://arxiv.org/abs/2401.14295) |

### Resources

- **Anthropic Prompt Engineering Docs**: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview
- **Anthropic Interactive Tutorial**: https://github.com/anthropics/prompt-eng-interactive-tutorial
- **Anthropic Cookbook (Agent Patterns)**: https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents
- **OpenAI Prompt Engineering Guide**: https://platform.openai.com/docs/guides/prompt-engineering
- **DAIR.AI Prompt Engineering Guide**: https://www.promptingguide.ai/
- **LLM Agents Papers (actively updated)**: https://github.com/AGI-Edgerunners/LLM-Agents-Papers
- **DSPy Documentation**: https://dspy.ai
- **RLM Implementation**: https://github.com/alexzhang13/rlm

---

## Key Takeaways

1. **The model is a simulator, not an entity.** Prompt for the perspective you want simulated, not for "the AI's opinion."

2. **More inference-time compute generally helps.** Whether through CoT, self-consistency, tree search, or recursion—giving models more "thinking time" improves hard problems.

3. **The techniques stack.** You can combine few-shot examples + CoT + role prompting. In code, you can add self-consistency on top of ReAct.

4. **Match technique to problem.** Simple questions don't need elaborate prompting. Save the heavy machinery for tasks that actually require it.

5. **Start simple, add complexity only when needed.** Anthropic's core guidance: optimized single calls often beat agents. Don't reach for complexity prematurely.

6. **The frontier is moving fast.** Reasoning models have internalized CoT. RLMs are internalizing context management. What required elaborate prompting scaffolds yesterday may be built into tomorrow's models.

7. **The best prompt is often a better problem definition.** As models get more capable, the bottleneck shifts from "how to make it think" to "what exactly do you want it to do."

---

*Last updated: January 2026*
*Compiled for practitioners building production AI systems*
