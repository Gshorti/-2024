from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM

dataset = load_dataset("IlyaGusev/ru_turbo_saiga", split='train')
print(dataset)
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

def formatting_prompts_func(example):
    # print(example['messages'])
    output_texts = []
    for i in example['messages']:
        print(i['content'])
        for j in range(len(i['content'][0:-1:2])):
            text = f"### Question: {i['content'][j]}\n ### Answer: {i['content'][j + 1]}"

            output_texts.append(text)
    return output_texts


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=16,
)
trainer.train()

trainer.save_model('modelforTechnoStrelkaNEW')
model.save_pretrained('modelforTechnoStrelkaNEW2')