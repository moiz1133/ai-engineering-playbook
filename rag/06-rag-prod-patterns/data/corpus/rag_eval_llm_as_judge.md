# LLM-as-Judge for RAG Evaluation

When evaluating generated answers at scale, using a large language model to grade another model's output — "LLM-as-judge" — has become the practical default, since human review doesn't scale to the volume of test cases most teams need to run continuously.

A typical LLM-judge setup gives the judge model the original question, the retrieved context, and the generated answer, then asks it to score along specific dimensions: faithfulness (does the answer only state things supported by the context, without hallucinating), relevance (does the answer actually address the question asked), and completeness (does it capture the key information present in the context, without unnecessary omission). Scoring each dimension separately, rather than asking for one overall quality score, produces more actionable and more consistent results, since a single blended score hides which dimension actually failed.

LLM judges are not perfectly reliable graders. They exhibit known biases: a preference for longer answers regardless of quality, sensitivity to the exact wording of the grading prompt, and some correlation with the judge model's own generation style — a judge model tends to rate answers that "sound like itself" more favorably. Using a stronger, different model as judge than the one generating answers (avoiding a model grading its own output) and providing explicit grading rubrics with examples of each score level both measurably improve judge reliability.

Because of these caveats, most teams treat LLM-judge scores as a relative signal for tracking regressions over time (did this change make things worse?) rather than as an absolute measure of quality, and periodically validate the judge itself against a small sample of human-graded answers to confirm it hasn't drifted from human judgment.

LLM-as-judge is particularly valuable for RAG specifically because faithfulness — whether an answer is actually grounded in the retrieved context rather than the model's parametric knowledge — is difficult to check with simple string-matching metrics but is exactly the kind of nuanced judgment a capable LLM can assess well.
