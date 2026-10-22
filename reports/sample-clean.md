<!-- Sample output, committed as an example. Not regenerated on each run. -->

> A clean run. The critic accepted the first draft, so this is what the pipeline produces with no revision.
>
> **Question:** How does QLoRA compare to full fine-tuning in downstream task performance?
>
> Sources: 6 · Drafts: 1 · Sourced claim rate: 100.0% · Citation coverage: 100.0%

---

## Summary

QLoRA achieves lower downstream task performance than full fine-tuning, with quality metrics 93–96% of full fine-tuning compared to LoRA's 93–97% [4]. Full fine-tuning remains the highest-performing approach but demands substantially more computational resources [1]. QLoRA offers significant memory efficiency gains—requiring 7.2 GB VRAM versus 56 GB for full fine-tuning on Llama-2-7B—at the cost of slower training speed and reduced accuracy [4].

## Findings

### Accuracy and Performance Trade-offs

Full fine-tuning achieves the highest accuracy and performance [1]. QLoRA has slightly lower accuracy than LoRA [1], with quality metrics of 93–96% relative to full fine-tuning, compared to LoRA's 95–97% [4]. LoRA achieves high accuracy with faster training than full fine-tuning [1].

### Resource Requirements

Full fine-tuning requires high GPU/memory resources [1]. Specifically, full fine-tuning requires 56 GB VRAM for Llama-2-7B, while LoRA requires 18 GB and QLoRA requires 7.2 GB [4]. QLoRA training speed is 10–20% slower than LoRA [4].

### Task-Dependent Performance

LoRA makes much faster and more efficient fine-tuning possible than full fine-tuning while still achieving comparable performance on many tasks [3]. However, LoRA has difficulty matching the performance of full fine-tuning on difficult tasks like code generation and long-form text generation [5]. On financial tasks specifically, LoRA methods achieved substantial performance gains of 36% on average over base models [6].

### Mechanistic Differences

LoRA and full fine-tuning yield weight matrices whose singular value decompositions exhibit very different structure [5]. Weight matrices trained with LoRA have new, high-ranking singular vectors called intruder dimensions, while those trained with full fine-tuning do not [5]. LoRA forgets less than full fine-tuning [5].

## Contradictions and Open Questions

**Comparability on downstream tasks:** One source states that LoRA achieves comparable performance to full fine-tuning on many tasks [3], while another reports that LoRA has difficulty matching full fine-tuning performance on difficult tasks like code generation and long-form text generation [5]. The sources do not clarify which downstream tasks fall into each category or provide direct QLoRA performance comparisons on these difficult tasks.

The sources provide limited direct comparisons of QLoRA to full fine-tuning on specific downstream tasks beyond the aggregate quality metrics cited above.

## Sources

[1] LLM Fine-Tuning — Full vs LoRA vs QLoRA | Key Differences & Use Cases | Uplatz — https://www.youtube.com/watch?v=tS7ERxLAmVU

[2] LLM Fine-Tuning: LoRA vs Full Fine-Tuning — a Comparison | by Florin Andrei | Towards AI — https://pub.towardsai.net/llm-fine-tuning-lora-vs-full-fine-tuning-a-comparison-3aa1c1a0dc4d

[3] Fine-Tuning using LoRA and QLoRA - GeeksforGeeks — https://www.geeksforgeeks.org/deep-learning/fine-tuning-using-lora-and-qlora

[4] LoRA vs QLoRA: A Practical Comparison for LLM Fine-Tuning | Brahim Bouine — https://www.brahimbouine.com/blog/llm-fine-tuning-lora-qlora-comparison

[5] LoRA vs Full Fine-tuning: An Illusion of Equivalence — https://arxiv.org/html/2410.21228v3

[6] FinLoRA: Benchmarking LoRA Methods for Fine-Tuning LLMs on Financial Datasets — https://arxiv.org/html/2505.19819v1