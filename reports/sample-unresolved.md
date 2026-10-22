<!-- Sample output, committed as an example. Not regenerated on each run. -->

> A run the critic never accepted. It raised problems across three drafts; finalize released the draft it rated best and attached the outstanding issues at the end.
>
> **Question:** What are the main technical approaches to detecting model drift in production ML systems?
>
> Sources: 6 · Drafts: 3 · Sourced claim rate: 78.6% · Citation coverage: 100.0%

---

## Summary

Detecting model drift in production ML systems involves monitoring three distinct types of changes: concept drift (shifts in learned data patterns), data drift (changes in feature distributions), and performance drift (correctness of predictions against ground truth) [1][3]. When ground truth labels are unavailable, practitioners use proxy metrics and statistical techniques to identify drift early [1][6].

## Findings

### Concept Drift Detection

Concept drift refers to changes in the data patterns and relationships that the ML model has learned, potentially causing a decline in production model quality [1]. When ground truth labels are available, model quality can be monitored directly using metrics like accuracy or mean error [1]. When labels are unavailable, proxy metrics including prediction drift, data drift, correlation changes, or domain-specific heuristics can detect concept drift [1].

### Data Drift Detection

Data drift refers to changes in the distribution of features an ML model receives in production, potentially causing a decline in model performance [6]. Data drift monitoring techniques serve as proxy signals to assess whether an ML system operates under familiar conditions when ground truth labels aren't accessible [6].

Data distribution drift can be detected using several technical approaches: monitoring summary feature statistics, statistical hypothesis testing, or distance metrics [6]. Three nonparametric methods are used for data drift detection: Hellinger Distance (HD), Janson-Shannon (JS) Divergence, and Kolmogorov-Smirnov (KS) Test [5]. Hellinger Distance and JS Divergence quantify the discrepancy between probability distributions, while the KS method provides a statistical framework that can assess changes in multivariate data [5]. The Kolmogorov Smirnov Statistic specifically detects differences in data distributions based on the maximum difference between cumulative distributions of a reference distribution and a recent distribution [4].

### Output and Performance Drift Detection

Output data drift (prediction results) can be monitored by tracking predicted distributions or the length of generated text as an early warning that the model may be experiencing drift [3]. Performance drift refers to whether a prediction was correct, determined by ground truth [3].

### Integration into Production Systems

Data drift detection techniques can be integrated into the classification pipeline to dynamically adapt to changing data environments and improve model performance [5].

## Contradictions and Open Questions

No contradictions were found in the sources.

The sources provide limited detail on implementation specifics, comparative performance of different methods, real-time latency requirements, or cost-benefit tradeoffs between approaches.

## Sources

[1] What is concept drift in ML, and how to detect and address it — https://www.evidentlyai.com/ml-in-production/concept-drift

[2] Data Drift: Key Detection and Monitoring Techniques in 2026 | Label Your Data — https://labelyourdata.com/articles/machine-learning/data-drift

[3] A Guide To ML Monitoring And Drift Detection — https://www.bentoml.com/blog/a-guide-to-ml-monitoring-and-drift-detection

[4] Model Drift Detection with Kolmogorov Smirnov Statistic on Spark | Mawazo — https://pkghosh.wordpress.com/2020/02/24/model-drift-detection-with-kolmogorov-smirnov-statistic-on-spark

[5] An Empirical Analysis of Data Drift Detection Techniques in ... — https://repositorio.usp.br/bitstreams/fe135f6e-a245-48c2-9390-e5c204eee4f1

[6] What is data drift in ML, and how to detect and handle it — https://www.evidentlyai.com/ml-in-production/data-drift

---

## Unresolved Review Issues

The revision budget was exhausted without the reviewer accepting. Draft 1 of 3 is released as the one with the fewest outstanding issues. Treat the affected claims with caution.

- The sentence 'The Kolmogorov Smirnov Statistic specifically detects differences in data distributions based on the maximum difference between cumulative distributions of a reference distribution and a recent distribution [4]' cites [4], but the verified claims show this statement comes from an unsupported source (marked with [?] in the verified claims). The claim about KS statistic is not verified by the provided sources.
- The report cites [4] for the KS statistic explanation, but [4] does not appear in the verified claims as a supporting source for that specific claim. This is a citation pointing to a source that does not support the claim per the notes.
