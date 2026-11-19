# OMEN --- Inference Intelligence Layer

## Product specification and technical operating model

**Document type:** Product and technical specification\
**Product:** OMEN\
**Layer:** Inference Intelligence\
**Status:** Proposed product design\
**Audience:** Product, engineering, data, and growth teams

------------------------------------------------------------------------

## 1. Executive summary

The Inference Intelligence layer turns OMEN from a basic GPU/model fit
calculator into a platform that helps developers and AI teams make
evidence-based decisions about deploying and operating large language
models (LLMs).

It answers five connected questions:

1.  **What can I run?** --- model, hardware, precision, memory, and
    engine compatibility.
2.  **How will it perform?** --- expected prefill and decode
    performance, latency, throughput, and concurrency.
3.  **What will it cost?** --- infrastructure cost and cost per unit of
    useful work.
4.  **What should I change?** --- ranked hardware, quantization, engine,
    and workload configurations.
5.  **How reliable is this answer?** --- evidence provenance, benchmark
    comparability, confidence, and prediction error.

The layer combines: - A normalized model and hardware catalog. - A
standardized real-world benchmark dataset. - A workload and
configuration model. - An estimation and calibration engine. -
Comparison, simulation, and recommendation workflows. - Transparent
evidence and uncertainty reporting.

**Product principle:** OMEN must clearly distinguish measured results
from estimates. A prediction must never be presented as a benchmark
result.

## 2. Product goals and non-goals

### 2.1 Goals

-   Help a developer determine whether a model configuration can fit on
    available hardware.
-   Compare models, GPUs, quantization formats, and inference engines
    under comparable conditions.
-   Estimate performance for a declared workload.
-   Show the assumptions and uncertainty behind every estimate.
-   Use verified benchmark data to calibrate and evaluate predictions.
-   Recommend configurations that satisfy explicit memory, latency,
    throughput, quality, and budget constraints.
-   Make benchmark data useful, discoverable, and reusable.
-   Build repeat usage through saved workloads, comparisons, reports,
    and updated benchmark data.

### 2.2 Non-goals for the initial release

The initial Inference Intelligence layer does not need to: - Host or
serve customers' production LLM traffic. - Orchestrate GPU fleets. -
Automatically purchase or provision cloud GPUs. - Guarantee performance
on hardware that has not been measured. - Claim universal accuracy from
a small benchmark sample. - Replace a full observability platform. -
Automatically change a customer's production deployment.

Those capabilities belong to later products or integrations.

## 3. Core concepts and terminology

  -----------------------------------------------------------------------
  Term                                Meaning
  ----------------------------------- -----------------------------------
  Model                               A specific model architecture and
                                      checkpoint/revision.

  Model configuration                 Model plus weights format,
                                      quantization, runtime options, and
                                      relevant settings.

  Hardware configuration              GPU(s), VRAM, interconnect, CPU,
                                      RAM, and relevant system details.

  Inference engine                    Software executing inference, such
                                      as vLLM or llama.cpp.

  Workload                            The request pattern: input/output
                                      lengths, concurrency, request rate,
                                      and service objectives.

  Prefill                             Processing prompt tokens and
                                      constructing attention state.

  Decode                              Autoregressively generating output
                                      tokens.

  TTFT                                Time to first generated token.

  TPOT                                Time per output token, after the
                                      first token; definitions must be
                                      standardized.

  Throughput                          Tokens or requests completed per
                                      unit of time. The token type and
                                      measurement window must be stated.

  Concurrency                         Number of simultaneously active
                                      requests.

  KV cache                            Stored key/value attention tensors
                                      used during autoregressive
                                      generation.

  Quantization                        Representing model weights and/or
                                      activations with lower-precision
                                      numerical formats.

  Benchmark run                       A reproducible execution with
                                      recorded configuration, workload,
                                      and measurements.

  Estimate                            A model-derived prediction, not a
                                      direct measurement.

  Verified benchmark                  A benchmark that passed OMEN's
                                      defined validation process.
  -----------------------------------------------------------------------

## 4. Inference Intelligence feature map

The layer is made up of the following product modules.

  -----------------------------------------------------------------------
  Module                  Main user question      Primary output
  ----------------------- ----------------------- -----------------------
  A. Model Intelligence   What model am I         Normalized model
                          evaluating?             profile

  B. Hardware             What hardware can run   Hardware profile and
  Intelligence            it?                     compatibility

  C. Configuration        Which runtime           Reproducible
  Intelligence            configuration matters?  configuration

  D. Benchmark            What has actually been  Comparable benchmark
  Intelligence            measured?               records

  E. Performance          How fast might it run?  Performance estimates
  Intelligence                                    and measured
                                                  comparisons

  F. Memory Intelligence  Will it fit, and at     Memory budget and
                          what                    capacity analysis
                          context/concurrency?    

  G. Workload             What will my            Workload profile and
  Intelligence            application demand?     simulation

  H. Cost Intelligence    What will it cost?      Cost model and scenario
                                                  comparison

  I. Comparison Studio    Which option is better? Side-by-side
                                                  comparisons

  J. Recommendation       What should I deploy?   Ranked feasible
  Engine                                          configurations

  K. Confidence and       How much should I trust Provenance,
  Evidence                this?                   uncertainty, and
                                                  prediction error

  L. Saved Workspaces     Can I revisit and share Saved projects and
                          the analysis?           reports

  M. Data Contribution    Can I improve the       Submitted benchmark
                          dataset?                records

  N. Alerts and Change    What changed since my   Updated recommendations
  Tracking                last analysis?          and alerts

  O. Developer API and    Can I use OMEN          API responses and
  Export                  programmatically?       machine-readable
                                                  reports
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 5. Detailed feature specifications

## A. Model Intelligence

### Purpose

Provide a normalized, searchable catalog of models and the technical
metadata needed for memory, performance, and compatibility analysis.

### Features

1.  **Model search and discovery**
    -   Search by model name, organization, family, task, parameter
        count, and architecture.
    -   Filter by text-generation capability, model family, size,
        license metadata, and supported format.
    -   Show model aliases and canonical identifiers.
2.  **Hugging Face model import**
    -   Accept a model repository identifier.
    -   Retrieve public metadata when available.
    -   Record the exact repository revision or commit.
    -   Parse configuration metadata such as architecture, layer count,
        hidden size, attention heads, KV heads, vocabulary size, and
        maximum context metadata.
    -   Retrieve weight-file metadata where accessible.
    -   Flag missing or unsupported metadata rather than silently
        guessing.
3.  **Model architecture profile**
    -   Architecture family.
    -   Parameter count and, where available, active parameter count.
    -   Number of layers.
    -   Hidden dimension.
    -   Attention head count and KV head count.
    -   Head dimension.
    -   Vocabulary size.
    -   Context-window metadata.
    -   Weight tying and relevant architecture properties.
    -   MoE experts and active experts, where applicable.
4.  **Model variants**
    -   Base and instruction-tuned variants.
    -   Fine-tunes.
    -   Distilled variants.
    -   Quantized releases.
    -   Different checkpoint revisions.
5.  **Model compatibility**
    -   Supported inference engines.
    -   Supported weight formats and quantization formats.
    -   Known architecture limitations.
    -   Required custom code or dependencies, if applicable.
    -   Hardware support caveats.
6.  **Model version history**
    -   Track revisions and metadata changes.
    -   Preserve the revision used by each benchmark.
    -   Avoid treating two different revisions as identical evidence.

### How it works

1.  A user searches for a model or enters a repository identifier.
2.  OMEN resolves the identifier to a canonical model record and
    revision.
3.  The ingestion service retrieves metadata from supported sources.
4.  A parser maps source-specific fields into OMEN's normalized schema.
5.  Validation checks required fields and plausible values.
6.  Missing fields are marked unknown; they are not invented.
7.  The model profile is stored and made available to the calculation
    and comparison engines.

### Important implementation detail

A model's advertised context length is not a promise that a given
hardware configuration can serve that context. Actual memory feasibility
also depends on KV-cache format, batch/concurrency, architecture,
runtime overhead, and engine behavior.

------------------------------------------------------------------------

## B. Hardware Intelligence

### Purpose

Maintain a reliable catalog of accelerators and system configurations.

### Features

1.  **GPU catalog**
    -   Vendor and product family.
    -   Exact GPU model.
    -   VRAM capacity.
    -   Memory type.
    -   Memory bandwidth.
    -   Compute capability or relevant accelerator generation.
    -   Supported numerical formats, where known.
    -   Interconnect capabilities.
    -   Power and form-factor metadata, where available.
2.  **Multi-GPU systems**
    -   Number of GPUs.
    -   Total physical VRAM.
    -   Per-device VRAM.
    -   Interconnect type and topology.
    -   Whether model parallelism is required.
    -   Communication and sharding caveats.
3.  **CPU and host-memory profile**
    -   CPU model and architecture.
    -   System RAM.
    -   Relevant memory bandwidth where known.
    -   CPU-offload capability.
    -   Host-to-device transfer considerations.
4.  **Cloud hardware catalog**
    -   Instance type.
    -   Accelerator count and type.
    -   VRAM per accelerator.
    -   Host RAM and CPU.
    -   Region and availability metadata where sourced.
    -   On-demand and other pricing categories, with source and
        timestamp.
5.  **Hardware comparison**
    -   VRAM.
    -   Bandwidth.
    -   Compute capability.
    -   Supported formats.
    -   Measured model performance.
    -   Cost and power where reliable data exists.
6.  **Hardware data provenance**
    -   Manufacturer specification, provider listing, community record,
        or verified measurement.
    -   Source URL or reference.
    -   Last checked timestamp.
    -   Confidence/verification status.

### How it works

1.  Hardware records are ingested from manufacturer documentation,
    cloud-provider catalogs, and curated sources.
2.  Units and naming conventions are normalized.
3.  Conflicting values are stored with provenance and resolved through
    an explicit source-priority policy.
4.  The user selects a GPU or cloud instance.
5.  The estimator uses its memory, bandwidth, compute, and compatibility
    metadata.
6.  The UI displays the source and timestamp for important
    specifications.

### Important implementation detail

Do not assume that two GPUs with the same VRAM have the same inference
performance. Memory bandwidth, compute throughput, supported kernels,
interconnect, power limits, and software support all matter.

------------------------------------------------------------------------

## C. Configuration Intelligence

### Purpose

Represent the complete set of choices that determine an inference run.

### Features

-   Model and exact revision.
-   Weight format and quantization.
-   Activation and KV-cache precision where supported.
-   Inference engine and version.
-   GPU model and GPU count.
-   Tensor/pipeline parallel settings.
-   Context length.
-   Maximum sequence length.
-   Batch and concurrency settings.
-   Prefix caching and other runtime options.
-   Speculative decoding settings where applicable.
-   CPU offload settings.
-   Attention implementation.
-   Container/runtime and relevant driver versions.

### How it works

A configuration is a versioned object, not a loose collection of UI
selections. It receives a stable identifier or hash so benchmark runs
and saved reports can refer to the exact same configuration.

When a user changes a material setting, OMEN creates a new configuration
revision. It must not merge results from materially different
configurations as though they were identical.

------------------------------------------------------------------------

## D. Benchmark Intelligence

### Purpose

Build a trusted, searchable dataset of real inference measurements.

This is one of OMEN's most important long-term data assets.

### Features

1.  **Benchmark catalog**
    -   Search by model, GPU, engine, quantization, context length, and
        workload.
    -   Filter by benchmark status and source.
    -   View the exact test configuration.
2.  **Standardized benchmark runner**
    -   Provide a reproducible benchmark harness.
    -   Support selected inference engines.
    -   Record warm-up and measured iterations.
    -   Capture input and output token counts.
    -   Capture TTFT, inter-token latency/TPOT, throughput, and peak
        memory where available.
    -   Capture hardware and software metadata.
    -   Export a JSON result.
3.  **Benchmark submission**
    -   Upload a result file or submit through a CLI.
    -   Require explicit consent before publishing.
    -   Validate schema and required metadata.
    -   Detect incomplete or suspicious records.
    -   Preserve submitter and source attribution according to privacy
        settings.
4.  **Benchmark validation**
    -   Schema validation.
    -   Hardware and model identity checks.
    -   Configuration consistency checks.
    -   Plausibility checks.
    -   Duplicate detection.
    -   Reproduction status.
    -   Optional reviewer approval.
5.  **Benchmark status**
    -   **Verified:** reproduced or checked under OMEN's controlled
        verification process.
    -   **Sourced:** imported from a traceable external publication or
        dataset.
    -   **Community:** user-submitted and not independently reproduced.
    -   **Estimated:** model prediction; never a benchmark record.
6.  **Benchmark comparison**
    -   Compare only compatible workloads or clearly disclose
        differences.
    -   Display sample count and variation.
    -   Show measurement methodology.
    -   Avoid ranking results based on incomparable token lengths or
        concurrency.
7.  **Benchmark history**
    -   Track engine and driver changes.
    -   Compare versions.
    -   Detect performance regressions or improvements.
    -   Retain historical records.

### How it works

1.  A contributor selects a supported model, engine, and hardware.
2.  The runner checks prerequisites and records the environment.
3.  It performs warm-up runs.
4.  It executes the defined workload.
5.  It captures metrics and metadata.
6.  It writes a structured result.
7.  The contributor reviews and explicitly submits the result.
8.  OMEN validates, deduplicates, and assigns a status.
9.  The benchmark becomes searchable according to its status.

### Benchmark protocol requirements

Each record should include, where applicable:

-   Model ID and revision.
-   GPU model and count.
-   Driver and accelerator runtime versions.
-   Inference engine and version.
-   Quantization and KV-cache precision.
-   Prompt/input token distribution.
-   Output token count or distribution.
-   Context length.
-   Batch size and concurrency.
-   Request rate and test duration.
-   Warm-up count.
-   TTFT definition and measurement method.
-   TPOT/inter-token latency definition.
-   Throughput definition: input tokens, output tokens, or total tokens
    per second.
-   Peak VRAM and measurement method.
-   Power measurement method, if collected.
-   Raw result or summary statistics.
-   Source, timestamp, and verification status.

### Important implementation detail

A benchmark leaderboard without workload normalization can mislead
users. For example, a result measured with short prompts and one active
request is not directly comparable to one measured with long contexts
and high concurrency.

------------------------------------------------------------------------

## E. Performance Intelligence

### Purpose

Estimate and explain inference performance for configurations that have
not necessarily been benchmarked.

### Features

1.  **Prefill performance estimation**
    -   Estimate prompt-processing performance.
    -   Account for prompt length, model architecture, hardware, and
        engine.
    -   Use measured calibration data when available.
2.  **Decode performance estimation**
    -   Estimate output-token generation speed.
    -   Consider weight memory traffic, memory bandwidth, batch size,
        KV-cache traffic, and kernel efficiency.
    -   Separate theoretical ceilings from calibrated estimates.
3.  **Latency estimation**
    -   TTFT estimate.
    -   Inter-token latency or TPOT estimate.
    -   End-to-end request latency estimate.
    -   Clearly state workload assumptions.
4.  **Throughput estimation**
    -   Output tokens per second.
    -   Input/prefill tokens per second.
    -   Requests per second where supported.
    -   Aggregate versus per-request throughput.
5.  **Concurrency analysis**
    -   Estimate how concurrency changes memory usage and throughput.
    -   Identify when latency objectives may no longer be met.
    -   Show queueing or saturation caveats.
6.  **Performance bottleneck analysis**
    -   Memory-bandwidth-bound.
    -   Compute-bound.
    -   KV-cache/memory-capacity-bound.
    -   Interconnect-bound.
    -   CPU/offload-bound.
    -   Runtime/kernel inefficiency.
    -   Insufficient evidence to classify.
7.  **Prediction-versus-measurement**
    -   Show estimated and measured values side by side.
    -   Calculate prediction error only for comparable conditions.
    -   Track error by model family, hardware family, engine, and
        workload.

### How it works

1.  The user selects a model, hardware, engine, quantization, and
    workload.
2.  OMEN checks whether an exact or near-exact benchmark exists.
3.  If a comparable benchmark exists, it is shown as measured evidence.
4.  Otherwise, the estimator calculates theoretical bounds and uses
    relevant calibration data.
5.  The system produces a range or point estimate with uncertainty.
6.  The UI explains which factors most influence the result.
7.  When a later benchmark becomes available, OMEN can compare it
    against the earlier prediction.

### Estimation methodology

Use multiple models rather than one universal formula.

For example:

-   Memory-capacity model for fit.
-   Memory-bandwidth model for a decode upper bound.
-   Compute model for prefill/compute limits.
-   KV-cache model for sequence and concurrency capacity.
-   Calibration model learned from compatible benchmark records.

The final prediction should be constrained by relevant bottlenecks and
calibrated only where evidence supports it.

Do not claim that a simple memory-bandwidth division predicts actual
tokens/sec for every model, GPU, engine, and workload.

------------------------------------------------------------------------

## F. Memory Intelligence

### Purpose

Determine whether a configuration fits in available device memory and
how much capacity remains for runtime state and concurrent requests.

### Features

1.  **Weight memory**
    -   Estimate model weight storage using parameter count and
        effective storage format.
    -   Account for quantization metadata/scales and format-specific
        overhead where known.
    -   Prefer actual checkpoint metadata when available.
2.  **KV-cache memory**
    -   Estimate from architecture, layers, KV heads, head dimension,
        sequence length, active sequences, and cache precision.
    -   Support architecture-specific formulas.
    -   Explain assumptions.
3.  **Runtime overhead**
    -   Account for framework, CUDA/runtime, attention workspace, graph
        capture, temporary buffers, and engine overhead when data is
        available.
    -   Avoid presenting a fixed universal overhead as exact.
4.  **Activation and workspace estimates**
    -   Model relevant runtime buffers based on engine and workload.
    -   Mark uncertain components.
5.  **Memory budget breakdown**
    -   Weights.
    -   KV cache.
    -   Activations/workspace.
    -   Runtime overhead.
    -   Reserved safety margin.
    -   Remaining headroom.
6.  **Context-length analysis**
    -   Show estimated memory at different context lengths.
    -   Identify the memory-limiting component.
7.  **Concurrency capacity**
    -   Estimate how many active sequences fit under a selected memory
        budget.
    -   Show that the answer also depends on scheduling, workload
        distribution, and latency constraints.
8.  **Multi-GPU memory analysis**
    -   Model sharding strategy.
    -   Per-GPU memory allocation.
    -   Replication versus sharding.
    -   Communication and uneven-shard overhead.
9.  **Offloading analysis**
    -   Explain CPU offload or partial offload when supported.
    -   Estimate transfer and performance implications only when
        adequate data exists.

### How it works

1.  Read the model architecture and selected weight format.
2.  Estimate or retrieve weight storage.
3.  Calculate KV-cache requirements for the declared context and active
    sequences.
4.  Add engine/runtime workspace estimates.
5.  Apply a configurable safety margin.
6.  Compare the total against usable device memory, not just advertised
    VRAM.
7.  Return fit status, memory breakdown, limiting factor, and caveats.

### Example output

  Memory component      Example estimate
  ------------------- ------------------
  Model weights                   9.0 GB
  KV cache                        3.0 GB
  Runtime/workspace               1.5 GB
  Safety reserve                  1.0 GB
  Total budget                   14.5 GB

These are illustrative numbers only.

The UI should distinguish **fits theoretically**, **likely fits**,
**measured to fit**, and **does not fit under stated assumptions**.

------------------------------------------------------------------------

## G. Workload Intelligence

### Purpose

Translate a real application workload into a reproducible inference
scenario.

### Features

1.  **Workload builder**
    -   Application type: chatbot, coding assistant, RAG, batch
        extraction, agent, or custom.
    -   Average and percentile input-token lengths.
    -   Average and percentile output-token lengths.
    -   Requests per second.
    -   Peak concurrency.
    -   Daily/monthly request volume.
    -   Traffic burstiness.
    -   Cache/prefix reuse assumptions.
2.  **Service-level objectives**
    -   Target TTFT.
    -   Target TPOT or inter-token latency.
    -   End-to-end latency target.
    -   Minimum throughput.
    -   Availability requirement (planning input, not a guarantee).
3.  **Traffic profiles**
    -   Steady traffic.
    -   Bursty traffic.
    -   Batch/offline traffic.
    -   Interactive traffic.
    -   Mixed workloads.
4.  **Workload templates**
    -   Customer support chatbot.
    -   RAG assistant.
    -   Code generation.
    -   Document processing.
    -   Agentic workflows.
    -   High-volume batch inference.
5.  **Workload simulation**
    -   Estimate memory demand.
    -   Estimate prefill and decode demand.
    -   Estimate throughput and latency under stated assumptions.
    -   Identify saturation and likely bottlenecks.
    -   Compare candidate configurations.
6.  **Workload import**
    -   Accept sanitized aggregate statistics or supported trace formats
        in a later release.
    -   Do not require raw prompts or personally identifiable data.
    -   Provide a local profiling path for sensitive workloads.

### How it works

1.  The user selects a template or enters workload parameters.
2.  OMEN validates the token-length and traffic assumptions.
3.  The workload is converted into a normalized workload object.
4.  The estimator evaluates candidate configurations.
5.  The simulator produces capacity, latency, and cost estimates.
6.  The user can save the workload and compare it across hardware or
    engines.

### Important implementation detail

Average input/output length is not enough for serious capacity planning.
Percentiles and distributions matter. A workload with a small average
but occasional very long contexts can have very different memory and
latency behavior.

------------------------------------------------------------------------

## H. Cost Intelligence

### Purpose

Estimate the financial cost of running a workload on candidate
infrastructure.

### Features

1.  **Cloud GPU pricing catalog**
    -   Provider and instance.
    -   GPU type and count.
    -   Region.
    -   Pricing model.
    -   On-demand, reserved, or spot/preemptible where supported.
    -   Source and last-updated timestamp.
2.  **Self-hosted cost model**
    -   Hardware acquisition cost.
    -   Amortization period.
    -   Power consumption and electricity price.
    -   Utilization assumption.
    -   Hosting/networking cost.
    -   Maintenance and operational assumptions.
    -   Exclude unknown costs or mark them as user inputs.
3.  **Cost metrics**
    -   Cost per hour.
    -   Cost per request.
    -   Cost per million input tokens.
    -   Cost per million output tokens.
    -   Cost per user or workload period.
    -   Total monthly cost.
4.  **Utilization analysis**
    -   Compare idle, average, and peak utilization.
    -   Show the effect of underutilized GPUs.
    -   Model capacity headroom.
5.  **Scenario comparison**
    -   Cloud versus self-hosted.
    -   One large GPU versus multiple smaller GPUs.
    -   Different quantization levels.
    -   Different concurrency targets.
    -   Different engine choices.
6.  **Break-even analysis**
    -   Estimate when owned hardware becomes cheaper than renting.
    -   Include amortization, utilization, power, hosting, and
        operational assumptions.
    -   Show sensitivity to utilization and token volume.
7.  **Price history**
    -   Track provider price changes.
    -   Preserve the price timestamp used in each saved report.

### How it works

1.  The workload module supplies expected token/request volume.
2.  The performance module estimates the infrastructure capacity
    required.
3.  The pricing catalog supplies current or user-entered prices.
4.  The cost engine calculates hourly and period costs.
5.  The system divides by useful work completed to calculate unit
    economics.
6.  The comparison view displays assumptions and sensitivity.

### Important implementation detail

Cost per token must account for actual utilization and useful
throughput. Dividing hourly GPU price by a theoretical peak token rate
can produce a misleadingly low cost.

------------------------------------------------------------------------

## I. Comparison Studio

### Purpose

Let users compare candidate deployment choices in one consistent view.

### Features

-   Model versus model.
-   GPU versus GPU.
-   Quantization versus quantization.
-   Engine versus engine.
-   Cloud provider versus provider.
-   Single-GPU versus multi-GPU.
-   Current deployment versus proposed deployment.
-   Workload-specific comparison.
-   Side-by-side memory, performance, cost, and compatibility.
-   Difference highlighting.
-   Shareable comparison links.
-   Export to CSV/JSON/PDF in later releases.

### How it works

1.  The user selects two or more candidate configurations.
2.  OMEN normalizes the workload and constraints.
3.  The system evaluates each configuration using the same assumptions.
4.  Measured data is shown separately from estimates.
5.  The UI highlights trade-offs and missing evidence.
6.  The user saves or shares the comparison.

### Comparison rules

-   Never compare results from different workloads without a clear
    warning.
-   Display the engine and version.
-   Display the measurement or estimate status.
-   Display the data timestamp.
-   Do not reduce every decision to one opaque score.

------------------------------------------------------------------------

## J. Recommendation Engine

### Purpose

Turn the collected evidence and workload requirements into a ranked
shortlist of feasible deployment configurations.

### Features

1.  **Constraint-based filtering**
    -   Budget ceiling.
    -   Maximum GPU count.
    -   Available VRAM.
    -   Required context length.
    -   Target throughput.
    -   Latency objectives.
    -   Required model and quality constraints.
    -   Cloud or on-premises preference.
2.  **Candidate generation**
    -   Hardware candidates.
    -   Quantization candidates.
    -   Engine candidates.
    -   Parallelism choices.
    -   Configuration combinations.
3.  **Feasibility checks**
    -   Memory fit.
    -   Engine/model compatibility.
    -   Hardware support.
    -   Workload capacity.
    -   Cost constraints.
    -   Evidence sufficiency.
4.  **Ranking modes**
    -   Lowest estimated cost.
    -   Highest throughput.
    -   Lowest latency.
    -   Best cost/throughput trade-off.
    -   Lowest memory requirement.
    -   Best evidence confidence.
5.  **Recommendation explanation**
    -   Why the configuration was selected.
    -   Which constraints it satisfies.
    -   Expected performance and cost.
    -   Main risks.
    -   What evidence supports it.
    -   What should be benchmarked before production use.
6.  **Alternative recommendations**
    -   Cheapest feasible option.
    -   Performance-oriented option.
    -   Balanced option.
    -   Lower-risk option with stronger benchmark evidence.
7.  **Recommendation validation**
    -   Allow users to benchmark the recommended configuration.
    -   Compare actual results with the recommendation.
    -   Feed prediction error into calibration.

### How it works

1.  The user defines a workload and constraints.
2.  The engine generates candidate configurations.
3.  Infeasible candidates are removed with a reason.
4.  Remaining candidates are evaluated for memory, performance, cost,
    and evidence.
5.  A transparent ranking policy orders candidates.
6.  The user receives explanations and alternatives.
7.  The chosen configuration can be saved and benchmarked.

### Important implementation detail

Do not use a single arbitrary weighted score as the only ranking
mechanism. First enforce hard constraints, then show Pareto trade-offs
between cost, latency, throughput, and confidence. Let users choose
their priorities.

------------------------------------------------------------------------

## K. Confidence, Evidence, and Prediction Quality

### Purpose

Make OMEN's results trustworthy and explain what is known, inferred, or
uncertain.

### Features

-   Evidence status labels.
-   Source citations and provenance.
-   Benchmark sample count.
-   Measurement variance.
-   Workload similarity score.
-   Hardware/configuration match level.
-   Prediction intervals where supported.
-   Prediction error tracking.
-   Calibration history.
-   Data freshness indicator.
-   Missing-data warnings.
-   Assumption disclosure.
-   Out-of-distribution warnings.

### Evidence hierarchy

1.  Exact configuration, same workload, verified measurement.
2.  Closely matched configuration and workload, verified measurement.
3.  Traceable external benchmark with comparable methodology.
4.  Calibrated estimate supported by related measurements.
5.  First-principles theoretical estimate.
6.  Insufficient evidence.

### How it works

1.  Every result is associated with its inputs and evidence.
2.  The system searches for matching benchmark records.
3.  It evaluates how closely each record matches the requested
    configuration and workload.
4.  It selects the appropriate evidence class.
5.  It calculates uncertainty only where enough data exists.
6.  It displays assumptions, caveats, and provenance with the result.

### Prediction evaluation

For a measured value (y) and prediction (`\hat{y}`{=tex}):

-   Absolute error: (\|y-`\hat{y}`{=tex}\|)
-   Absolute percentage error: (\|y-`\hat{y}`{=tex}\|/\|y\|
    `\times 100`{=tex}%), where percentage error is meaningful.
-   Track error distributions by metric, hardware, model family, engine,
    and workload.

Do not report a single global accuracy number if performance differs
materially across categories.

------------------------------------------------------------------------

## L. Saved Workspaces and Reports

### Purpose

Turn one-time calculator use into a recurring workflow.

### Features

-   User accounts.
-   Personal workspace.
-   Organization/team workspace.
-   Saved models and configurations.
-   Saved workloads.
-   Saved comparisons.
-   Saved recommendations.
-   Project folders and tags.
-   Notes and assumptions.
-   Version history.
-   Duplicate configuration.
-   Shareable read-only links.
-   Access control for private workspaces.
-   Exportable reports.

### How it works

1.  A user signs in.
2.  They create a project or workspace.
3.  They save a configuration, workload, or comparison.
4.  OMEN stores the exact input versions and evidence references.
5.  The user can reopen and update the analysis.
6.  Shared links expose only the data permitted by the workspace's
    access settings.

### Retention value

Users return when: - A new model is released. - GPU prices change. -
Their workload grows. - They are evaluating a new deployment. - A
benchmark changes the recommendation. - Their current configuration
stops meeting its service objectives.

------------------------------------------------------------------------

## M. Data Contribution and Community

### Purpose

Let the developer community expand OMEN's benchmark coverage.

### Features

-   Benchmark CLI.
-   Web-based result submission.
-   JSON upload.
-   Contributor profiles.
-   Attribution preferences.
-   Submission validation.
-   Duplicate detection.
-   Review queue.
-   Verified contributor status.
-   Public benchmark pages.
-   Contribution history.
-   Dataset export under a clearly defined license.

### How it works

1.  A developer runs the OMEN benchmark tool locally.
2.  The tool records configuration and results.
3.  The developer reviews the data.
4.  The developer explicitly chooses whether to submit it.
5.  OMEN validates and classifies the record.
6.  Approved records appear in public comparisons.
7.  Contributors receive attribution if they opt in.

### Anti-abuse controls

-   Rate limits.
-   Duplicate detection.
-   Plausibility checks.
-   Provenance requirements.
-   Reproduction checks.
-   Moderation and dispute handling.
-   Separate unverified and verified data.
-   No silent overwriting of benchmark history.

------------------------------------------------------------------------

## N. Alerts and Change Tracking

### Purpose

Notify users when changes could affect a saved deployment decision.

### Features

-   GPU price changes.
-   Cloud instance price changes.
-   New benchmark results.
-   Model revision changes.
-   Inference engine version changes.
-   Performance regression alerts.
-   Updated prediction calibration.
-   Saved workload feasibility changes.
-   Recommendation changes.

### How it works

1.  A saved project references models, hardware, prices, benchmarks, and
    assumptions.
2.  Background jobs check relevant source updates.
3.  OMEN identifies material changes.
4.  The system recalculates affected analyses.
5.  It records the before/after difference.
6.  The user receives an in-app or email notification according to their
    settings.

Alerts must explain what changed and why it matters. Avoid sending
notifications for immaterial metadata changes.

------------------------------------------------------------------------

## O. Developer API and Export

### Purpose

Allow developers and internal tools to access OMEN's intelligence
programmatically.

### Features

-   Model lookup API.
-   GPU lookup API.
-   Compatibility API.
-   Memory estimation API.
-   Performance estimation API.
-   Workload simulation API.
-   Cost estimation API.
-   Benchmark search API.
-   Configuration comparison API.
-   Recommendation API.
-   API keys and usage limits.
-   Versioned response schemas.
-   Usage analytics.
-   JSON and CSV export.

### Example API design

These are proposed endpoints, not currently implemented endpoints.

``` http
GET  /api/v1/models/{model_id}
GET  /api/v1/gpus/{gpu_id}
POST /api/v1/estimate/memory
POST /api/v1/estimate/performance
POST /api/v1/workloads/simulate
POST /api/v1/cost/estimate
POST /api/v1/configurations/compare
POST /api/v1/recommendations
GET  /api/v1/benchmarks
POST /api/v1/benchmarks/submit
```

### Example request

``` json
{
  "model": {
    "id": "organization/model-name",
    "revision": "commit-or-revision"
  },
  "hardware": {
    "gpu": "GPU_MODEL",
    "count": 1
  },
  "configuration": {
    "quantization": "int4",
    "engine": "vllm"
  },
  "workload": {
    "input_tokens_p50": 2048,
    "output_tokens_p50": 512,
    "concurrency": 8
  }
}
```

### Example response shape

``` json
{
  "status": "estimated",
  "memory": {
    "estimated_total_gb": null,
    "confidence": "insufficient_data"
  },
  "performance": {
    "ttft_ms": null,
    "output_tokens_per_second": null
  },
  "evidence": {
    "type": "model_estimate",
    "benchmark_count": 0
  },
  "warnings": [
    "No comparable verified benchmark is available."
  ]
}
```

The null values are intentional: the API must not fabricate a result
when the necessary data or calibration is unavailable.

------------------------------------------------------------------------

# 6. End-to-end user workflows

## Workflow 1: "Can I run this model on my GPU?"

1.  User opens the model catalog.
2.  Selects a model or imports a Hugging Face repository.
3.  Selects their GPU and quantization.
4.  Enters context length and concurrency.
5.  Memory Intelligence calculates the memory budget.
6.  Compatibility checks validate model, engine, and hardware support.
7.  Confidence module checks whether similar configurations have been
    measured.
8.  OMEN returns fit status, memory breakdown, assumptions, and
    alternatives.
9.  User saves or shares the configuration.

## Workflow 2: "Which GPU should I buy?"

1.  User selects the model and workload.
2.  Enters budget and latency/throughput requirements.
3.  OMEN generates candidate GPUs and instances.
4.  Memory Intelligence removes infeasible candidates.
5.  Performance Intelligence estimates the remaining candidates.
6.  Cost Intelligence calculates total and unit costs.
7.  Comparison Studio shows the trade-offs.
8.  Recommendation Engine ranks feasible options.
9.  User receives a report with evidence and caveats.

## Workflow 3: "How much will my application cost?"

1.  User creates a workload profile.
2.  Enters monthly request volume, token distributions, and peak
    traffic.
3.  Selects candidate deployment configurations.
4.  Performance Intelligence estimates required capacity.
5.  Cost Intelligence calculates infrastructure cost.
6.  Scenario analysis compares cloud and self-hosted options.
7.  User saves the financial model and assumptions.

## Workflow 4: "Is my deployment inefficient?"

This is a later-stage workflow requiring measured customer data.

1.  User imports sanitized aggregate metrics or runs a local profiler.
2.  OMEN receives token volume, latency, utilization, and infrastructure
    details.
3.  The system identifies bottlenecks and unused capacity.
4.  It compares the current deployment against candidate configurations.
5.  It estimates potential savings and performance changes.
6.  User reviews the evidence and exports a recommendation.
7.  User validates the proposed change in their own environment.

OMEN should not claim guaranteed savings without a controlled
comparison.

------------------------------------------------------------------------

# 7. Technical architecture

## 7.1 Recommended initial architecture

Start with a modular monolith, not microservices.

``` text
OMEN Web App
    |
    v
FastAPI Application
    |
    +-- Model Catalog Module
    |
    +-- Hardware Catalog Module
    |
    +-- Configuration Module
    |
    +-- Memory Estimation Module
    |
    +-- Performance Estimation Module
    |
    +-- Workload Simulation Module
    |
    +-- Cost Estimation Module
    |
    +-- Benchmark Module
    |
    +-- Comparison Module
    |
    +-- Recommendation Module
    |
    +-- Workspace and Auth Module
    |
    v
PostgreSQL
    |
    +-- Redis (cache and background jobs)
    |
    +-- Object Storage (benchmark artifacts and reports)
```

## 7.2 Suggested stack

  -----------------------------------------------------------------------
  Layer                   Technology              Role
  ----------------------- ----------------------- -----------------------
  Frontend                React + TypeScript      Interactive application

  Styling                 Tailwind CSS            UI

  Backend                 FastAPI                 APIs and business logic

  Database                PostgreSQL              Catalogs, benchmarks,
                                                  users, projects

  Cache                   Redis                   Cache and queue support

  Background jobs         Celery or Dramatiq      Ingestion and
                                                  recalculation

  Model metadata          Hugging Face Hub APIs   Model metadata
                                                  ingestion

  Benchmarking            vLLM and llama.cpp      Supported real-world
                                                  tests

  ML/calibration          Python, NumPy,          Estimation calibration
                          scikit-learn initially  

  Deployment              Docker                  Reproducible services

  Monitoring              Prometheus and Grafana  Service monitoring

  Authentication          Supabase Auth or        Identity
                          equivalent              

  Payments                Razorpay or Stripe,     Billing
                          subject to market       
                          availability            
  -----------------------------------------------------------------------

## 7.3 Core backend services/modules

### Catalog module

Owns model, GPU, cloud-instance, engine, and pricing metadata.

### Configuration module

Creates immutable/versioned inference configurations.

### Benchmark module

Ingests, validates, stores, searches, and compares measured results.

### Estimation module

Calculates memory, performance, workload capacity, and cost.

### Recommendation module

Generates candidates, applies hard constraints, and ranks feasible
configurations.

### Workspace module

Manages saved projects, access control, sharing, and reports.

### Background jobs

Refreshes catalogs, processes submissions, recalculates saved analyses,
and sends alerts.

------------------------------------------------------------------------

# 8. Suggested database entities

  -----------------------------------------------------------------------
  Entity                              Important fields
  ----------------------------------- -----------------------------------
  models                              id, canonical_name, architecture,
                                      revision, metadata_source

  model_revisions                     model_id, revision, config_hash,
                                      created_at

  gpus                                id, vendor, name, vram_bytes,
                                      bandwidth, source

  cloud_instances                     provider, region, instance_type,
                                      gpu_count, price

  engines                             name, version,
                                      supported_architectures

  configurations                      model_revision_id, gpu_id,
                                      engine_id, quantization,
                                      settings_hash

  workloads                           input_distribution,
                                      output_distribution, request_rate,
                                      concurrency, SLOs

  benchmark_runs                      configuration_id, workload_id,
                                      environment, status, source

  benchmark_metrics                   benchmark_run_id, metric_name,
                                      value, unit, aggregation

  pricing_records                     provider, instance_id, price,
                                      currency, timestamp

  estimates                           configuration_id, workload_id,
                                      outputs, assumptions,
                                      estimator_version

  recommendations                     workload_id,
                                      candidate_configurations,
                                      ranking_policy

  users                               identity and account metadata

  organizations                       team metadata

  projects                            organization_id, owner_id, project
                                      metadata

  saved_analyses                      project_id, input_snapshot,
                                      output_snapshot

  subscriptions                       organization_id, plan, billing
                                      status
  -----------------------------------------------------------------------

Store large raw artifacts in object storage and keep references and
metadata in PostgreSQL.

Use explicit units in the schema. For example, store bytes rather than
ambiguous "GB" values, and store latency in a consistent base unit.

------------------------------------------------------------------------

# 9. Data quality and governance rules

These rules should be enforced from the beginning.

1.  Never mix estimated and measured values without labels.
2.  Never discard benchmark configuration metadata.
3.  Never overwrite historical benchmark results.
4.  Preserve model revisions and engine versions.
5.  Record source and timestamp for external specifications and prices.
6.  Do not infer missing hardware or workload values silently.
7.  Show uncertainty where the estimator is weakly calibrated.
8.  Do not compare incompatible benchmark workloads as if they were
    equivalent.
9.  Obtain explicit consent before publishing user-submitted results.
10. Avoid collecting raw prompts when aggregate workload statistics are
    sufficient.
11. Provide a deletion and data-retention policy for user data.
12. Keep private organization benchmarks inaccessible to other
    organizations.

------------------------------------------------------------------------

# 10. Product UI / navigation structure

Suggested application navigation:

``` text
OMEN
|
+-- Overview
|
+-- Explore
|   +-- Models
|   +-- GPUs
|   +-- Inference Engines
|   +-- Cloud Instances
|
+-- Intelligence
|   +-- Model Fit
|   +-- Memory Analysis
|   +-- Performance Estimator
|   +-- Workload Simulator
|   +-- Cost Intelligence
|   +-- Recommendations
|
+-- Benchmarks
|   +-- Benchmark Explorer
|   +-- Leaderboards
|   +-- Compare Results
|   +-- Submit Benchmark
|
+-- Compare
|   +-- Model Comparison
|   +-- GPU Comparison
|   +-- Configuration Comparison
|
+-- Workspaces
|   +-- Saved Configurations
|   +-- Saved Workloads
|   +-- Reports
|
+-- Developer
|   +-- API Keys
|   +-- API Documentation
|   +-- CLI
|
+-- Settings
    +-- Profile
    +-- Organization
    +-- Billing
    +-- Privacy
```

For the first release, expose only the features that are actually
implemented. Do not show empty navigation sections just to imply product
maturity.

------------------------------------------------------------------------

# 11. Delivery roadmap

## Release 1 --- Intelligence foundation

-   Normalize model and GPU metadata.
-   Add model import.
-   Version configurations.
-   Improve memory estimates.
-   Add evidence labels.
-   Add basic GPU/model comparison.
-   Document calculation assumptions.

**Exit criteria:** Results are explainable, reproducible, and correctly
labeled.

## Release 2 --- Benchmark intelligence

-   Create benchmark schema.
-   Build benchmark CLI or runner.
-   Support a small number of engines and models first.
-   Add submission and validation.
-   Create benchmark explorer.
-   Add measured-versus-estimated comparisons.

**Exit criteria:** OMEN can collect and display comparable measurements
with provenance.

## Release 3 --- Workload and cost intelligence

-   Build workload builder.
-   Add percentile-based input/output distributions.
-   Add latency and throughput objectives.
-   Add cloud pricing catalog.
-   Add cost scenarios.
-   Add workload-specific configuration comparisons.

**Exit criteria:** Users can compare feasible deployment options for a
declared workload.

## Release 4 --- Recommendation and workspace

-   Add constraint-based candidate generation.
-   Add transparent ranking policies.
-   Add recommendation explanations.
-   Add saved workspaces and reports.
-   Add shareable links.
-   Add user accounts.

**Exit criteria:** Users can revisit and share deployment decisions.

## Release 5 --- SaaS and API

-   Add paid usage limits.
-   Add team workspaces.
-   Add API keys.
-   Add billing.
-   Add alerts and change tracking.
-   Add organization-level privacy and access controls.

**Exit criteria:** The product supports a paid recurring workflow.

------------------------------------------------------------------------

# 12. MVP prioritization

Do not build every feature simultaneously.

  Feature                       Initial priority   Reason
  ----------------------------- ------------------ ---------------------------------
  Model catalog/import          P0                 Required input
  GPU catalog                   P0                 Required input
  Configuration versioning      P0                 Reproducibility
  Memory Intelligence           P0                 Core fit decision
  Evidence labels               P0                 Trust
  Benchmark schema              P0                 Foundation for data moat
  Benchmark explorer            P1                 User retention
  GPU/model comparison          P1                 Acquisition and utility
  Performance Intelligence      P1                 Core differentiated value
  Workload builder              P1                 Makes estimates practical
  Cost Intelligence             P1                 Commercial value
  Recommendation Engine         P2                 Requires reliable upstream data
  Workspaces                    P2                 Retention
  Public API                    P2                 Developer distribution
  Alerts                        P3                 Advanced retention
  Enterprise private datasets   P3                 Enterprise expansion

**Recommended first public version:** Model Intelligence + Hardware
Intelligence + Memory Intelligence + Benchmark Explorer + basic
Performance Intelligence + Comparison Studio.

------------------------------------------------------------------------

# 13. Success metrics

Measure product usefulness, data quality, and commercial potential
separately.

### Product metrics

-   Percentage of users completing a model-fit analysis.
-   Time to first useful result.
-   Comparison creation rate.
-   Saved analysis rate.
-   Weekly and monthly returning users.

### Data metrics

-   Number of benchmark records.
-   Percentage with complete metadata.
-   Percentage independently verified.
-   Number of model/GPU/engine combinations covered.
-   Prediction error on held-out comparable benchmark runs.
-   Benchmark freshness.

### Recommendation metrics

-   Percentage of recommendations that satisfy hard constraints.
-   Percentage of recommendations supported by measured evidence.
-   Actual versus predicted performance after validation.
-   User acceptance rate.
-   User-reported usefulness.

### Business metrics

-   Free-to-paid conversion.
-   Paid workspace retention.
-   Monthly recurring revenue.
-   Churn.
-   Customer acquisition cost.
-   Customer-reported infrastructure savings, with methodology.

Avoid using raw benchmark count as the only measure of dataset quality.
A smaller set of well-described, reproducible measurements is more
valuable than a large set of incomparable numbers.

------------------------------------------------------------------------

# 14. Final product definition

The Inference Intelligence layer should become the analytical core of
OMEN.

It should allow a developer or AI team to:

1.  Discover and inspect a model.
2.  Select hardware and an inference engine.
3.  Determine memory feasibility.
4.  Define a realistic workload.
5.  Inspect measured performance and estimates.
6.  Compare candidate configurations.
7.  Calculate infrastructure cost.
8.  Receive explainable recommendations.
9.  Save and share the decision.
10. Validate the recommendation against actual benchmark results.

The long-term defensible asset is not the calculator interface. It is
the combination of:

-   Normalized model/hardware/configuration data.
-   High-quality real-world benchmark measurements.
-   A calibrated estimation engine.
-   Workload-aware recommendation logic.
-   A developer community contributing evidence.

**Product positioning:** OMEN Inference Intelligence --- the
evidence-based platform for understanding, comparing, and optimizing LLM
inference deployments.

------------------------------------------------------------------------

## 15. Explicit assumptions and open decisions

The following are product-design recommendations, not claims about
features already implemented in OMEN.

Before implementation, decide:

-   Which model families are supported first.
-   Which GPU vendors and hardware generations are in scope.
-   Which inference engines are supported in the first benchmark runner.
-   Which benchmark workload protocol is canonical.
-   Which external data sources may be ingested and under what licenses.
-   Whether benchmark submissions are public by default or opt-in.
-   Which metrics are required for a benchmark to be comparable.
-   Which cloud providers and regions are included in cost estimates.
-   Which estimator outputs are reliable enough to expose publicly.
-   What evidence threshold is required before a recommendation is
    labeled high confidence.

**Build rule:** Ship a smaller number of accurate, transparent,
reproducible intelligence features before expanding the catalog or
claiming broad prediction coverage.
