In order of Importance: 

- [X] Link between Problem & Solution for Idea
- [X] Try it yourself 
- [X] Improve "The Finding" 


- [ ] Literature Review
- [ ] Clean up repo for testing
- [ ] Header UI.


W/ Citations: 
Metacognition is a critical component of human intelligence. We reflect on our own thought processes, identify errors, and adapt our reasoning. 
En route to AGI, we need models that can do the same - to recognize when they're wrong and fix the error accordingly. **But how do we rigorously test this?**

Existing self-error detection benchmarks [citations] suffer from three primary limitations. First, they rely on human annotation, which is expensive, difficult to scale, and susceptible to model subjectivity bias [citation]. Second, they utilize heavily trained domains like mathematics, allowing models to predict common errors via pattern-matching rather than genuinely verifying reasoning steps. Third, they typically feature cascading errors where a flawed intermediate step leads to an incorrect final answer. Our preliminary experiments confirm this allows models to "hack" the evaluation: rather than tracing an error to its source, a model can simply re-derive the answer from scratch, compare final outcomes, and declare an error without actually locating it.

To overcome these limitations, we introduce Emoji-Bench, a benchmark of 1,000 novel logic questions designed to isolate and test true step-level verification. First, Emoji-Bench utilizes procedural generation for deterministic evaluation, eliminating the need for human annotators. Second, it employs novel formal systems represented by emojis, ensuring the tasks are entirely absent from training corpora and thus immune to pattern-matching. Finally, each question contains exactly one intermediate rule violation but is explicitly engineered to still reach the correct final answer. This renders re-execution and final-answer comparison useless, ensuring that models can only succeed through rigorous, step-by-step verification.

References: 
- RealMistake: https://arxiv.org/pdf/2404.03602
- CCUE: https://www.kaggle.com/benchmarks/tasks/anthonymbadiweikeme/ccue-task-2-self-error-detection