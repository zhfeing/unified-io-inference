from transformers import T5Tokenizer


tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=256, extra_ids=1100)

input_texts = [
    "what color is the sofa?",
    "/c22940/zhf/zhfeing_cygpu1/project/unified-io-inference/test/test_t5.py",
    "torchtext provides SOTA pre-trained models that can be used directly for",
    "NLP tasks or fine-tuned on downstream tasks. Below we use the pre-trained T5 model with standard",
    "base configuration to perform text summarization, sentiment classification, and translation. For additional details on available pre-trained models, please refer to documentation at https://pytorch.org/text/main/models.html"
]

output = tokenizer(
    input_texts,
    max_length=64,
    truncation=True,
    padding='longest'
)["input_ids"]

print(output)

