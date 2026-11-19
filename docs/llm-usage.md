# LLM usage statement

LLM-assisted coding, writing, and analysis are becoming standard in both industry
and research. While these tools entail risks, we believe that, combined with
careful verification of all generated content, they can accelerate scientific
progress. This, however, requires uncompromising transparency about which tools
contributed to the results and to what extent:

- **Implementation and documentation:** Claude (Anthropic) and Codex (OpenAI)
  were used as assistants for implementing, refactoring, and debugging parts of
  the framework, and for writing and editing the README, this documentation
  and the code comments. All code and documentation were reviewed and validated
  by the maintainers.
- **Design and experiments:** The architecture, the method, and all experimental
  decisions originate from the maintainers.
- **Models inside the pipeline:** Pretrained models that the framework calls at
  runtime (e.g., the LBM harmonization model) are components of the method, not
  development assistance.

If you find an error, please open an
[issue](https://github.com/Pfleiderer-Adrian/HybridSampleGeneration/issues). Thank you!!
