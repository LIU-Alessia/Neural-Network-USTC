"""
用预训练的Bert模型微调IMDB数据集，并使用SwanLabCallback回调函数将结果上传到SwanLab。
IMDB数据集的1是positive，0是negative。
使用多GPU并行训练
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from swanlab.integration.transformers import SwanLabCallback
import swanlab
import os

# 检查可用GPU数量
num_gpus = torch.cuda.device_count()
print(f"发现 {num_gpus} 个可用的GPU")

# 加载IMDB数据集
dataset = load_dataset('./imdb')

# 加载预训练的BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased')

# 定义tokenize函数
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# 对数据集进行tokenization
tokenized_datasets = dataset.map(tokenize, batched=True)

# 设置模型输入格式
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# 加载预训练的BERT模型
model = AutoModelForSequenceClassification.from_pretrained('./bert-base-uncased', num_labels=2)
# model = AutoModelForSequenceClassification.from_pretrained('./sentiment_model', num_labels=2)


# 如果有多块GPU，使用DataParallel包装模型
if num_gpus > 1:
    print(f"使用 {num_gpus} 个GPU进行并行训练")
    model = torch.nn.DataParallel(model)

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_first_step=100,
    # 总的训练轮数
    num_train_epochs=2,
    weight_decay=0.01,
    report_to="none",
    # 启用多GPU训练
    dataloader_num_workers=4 if num_gpus > 1 else 0,
    fp16=torch.cuda.is_available(),  # 如果GPU支持混合精度训练，则启用
)

CLASS_NAME = {0: "negative", 1: "positive"}

# 设置swanlab回调函数
swanlab_callback = SwanLabCallback(project='BERT',
                                   experiment_name='BERT-IMDB',
                                   config={
                                       'dataset': 'IMDB', 
                                       "CLASS_NAME": CLASS_NAME,
                                       "num_gpus": num_gpus,
                                       "device": str(device),
                                   })

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    callbacks=[swanlab_callback],
)

# 训练模型
trainer.train()

# 保存模型
# 如果是多GPU训练，保存时需要获取原始模型
if num_gpus > 1:
    model_to_save = model.module if hasattr(model, 'module') else model
else:
    model_to_save = model

model_to_save.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')

#评估模型
eval_results = trainer.evaluate(tokenized_datasets["test"])
print('------------验证集结果------------')
print(f"Eval results: {eval_results}")
# print(f"其他指标: {eval_results}")

# 测试模型
def predict(text, model, tokenizer, CLASS_NAME):
    inputs = tokenizer(text, return_tensors="pt")
    
    # 将输入移动到与模型相同的设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()

    print(f"Input Text: {text}")
    print(f"Predicted class: {int(predicted_class)} {CLASS_NAME[int(predicted_class)]}")
    return int(predicted_class)

test_reviews = [
    "I didn't enjoy this movie at all. It was confusing and the pacing was off. Definitely not worth watching."
]

model.eval()

text_list = []
for review in test_reviews:
    label = predict(review, model, tokenizer, CLASS_NAME)
    text_list.append(swanlab.Text(review, caption=f"{label}-{CLASS_NAME[label]}"))

if text_list:
    swanlab.log({"predict": text_list})

swanlab.finish()