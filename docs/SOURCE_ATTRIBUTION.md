# Source attribution

The public course starter is [AbDu11aHHH/ATML-PA2-LLM-PostTraining](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining), pinned at `05543bec4fba95eba708eae9d6e15730d53ca317`. The TA prompt-preserving preprocessing patch is pinned at `076e7cfdc7cd2f4b5e8e9e12cf15ef3652bc92da`. Reused infrastructure includes model/LoRA loading conventions, tokenization/chat templates, data plumbing, generation/scoring helpers, word counting, course configurations and fixed assets. The original executed source snapshots are retained in the evidence folders.

Student implementation, with Codex assistance, completes the DPO training/evaluation pipeline, corrects the objective and accuracy, preserves frozen-reference isolation and exact budgets, records reproducibility evidence, and adds the approved preprocessing/numerical compatibility handling. Publication-only additions are listed in [the audit](task1/AUDIT.md). Other task folders remain the supplied scaffolds.

Data: [course assets](https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets), revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`. They contain the fixed course UltraFeedback subsets; the student did not create or rebalance them. The ten fixed word-limit prompts are tracked in the starter and retained unchanged. Dataset files are not redistributed in this package beyond those ten prompts; generated outputs and metrics are preserved as evidence.

Policy/tokenizer: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Reward model: [yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback](https://huggingface.co/yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback), revision `f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc`. Their existing terms apply; no new license is asserted over third-party materials.

Research references:

- Rafailov et al. (2023), [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), for the objective and reference-policy interpretation.
- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 3.2, for the neural-attention definition underlying the qualitative relevance judgment.
- PyTorch gradient-scaling documentation is referenced in the recorded notebook and numerical reasoning. General scaling behavior is background, not proof of this run's exact overflow cause.
- The assignment manual defines the required experiment and metric conventions; TA clarifications here are user-supplied messages, not text silently inserted into the manual.

Codex/LLM assistance covered implementation, explanations, debugging, audit and documentation. The student ran the experiments in Colab, approved discretionary choices and qualitative examples, and is responsible for understanding/reviewing the submitted work. Historical code and failed attempts are retained rather than presented as newly authored or successful results.
