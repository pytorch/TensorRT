# Edge runtime input payloads

These JSON files are templates for the TensorRT-Edge-LLM C++ runners:

- `alpamayo_input_action.json` → `action_inference`
- `groot_input.json` → `llm_inference_groot`

Replace every `/path/to/...` image path with an absolute path visible to the
runtime process or container. For accuracy testing, replace the example state,
trajectory, images, prompt, generation settings, and seed with values from the
same sample used by eager evaluation.
