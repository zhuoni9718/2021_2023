def generate_answer(model, tokenizer, question, choices):
    inputs = [f"{question} {choice}" for choice in choices]
    input_ids = tokenizer(inputs, return_tensors="pt", padding=True).input_ids
    outputs = model.generate(input_ids)
    decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return decoded_outputs

def test_model(model_name, tokenizer_class, test_data):
    model = load_finetuned_model(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)

    correct = 0
    total = len(test_data)

    for data in test_data:
        question = data["data"]["question"]["stem"]
        choices = [choice["text"] for choice in data["data"]["question"]["choices"]]
        generated_answers = generate_answer(model, tokenizer, question, choices)
        answer_index = generated_answers.index(data["data"]["answerKey"])

        if answer_index == data["data"]["answerKey"]:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")

# 使用以下代码进行测试
model_name = 'facebook/bart-large'  # 更改为想要测试的模型名称
tokenizer_class = MODEL_CLASSES[model_name][1]
test_model(model_name, tokenizer_class, test_data)
