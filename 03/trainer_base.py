from modules.default_loader import DefaultLoader
from os.path import abspath, join
from os import environ
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix

class Main():
    def __init__(self) -> None:
        environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        self.TRAINDATAPATH = ["../data/text.train.txt", "../data/label.train.txt"]
        self.DEVDATAPATH = ["../data/text.dev.txt", "../data/label.dev.txt"]
        self.TESTDATAPATH = "../data/text.test.txt"
        
        self.RESULTPATH = "./result"
        
        self.checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        self.loader = DefaultLoader()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def main(self) -> None:
        dataset = self.preprocessing()
        self.train(dataset)
        

    def preprocessing(self) -> object:
        #? データ読み込み
        traindf = self.loader.loaddf(*self.TRAINDATAPATH)
        traindf = self.chenge_label_range(traindf)
        print(traindf)
        devdf = self.loader.loaddf(*self.DEVDATAPATH)
        devdf = self.chenge_label_range(devdf)
        print(devdf)
        testdf = self.loader.loadtestdf(self.TESTDATAPATH)
        print(testdf)
        
        #? ここで前処理
        
        #? データセットに変換
        train_dataset = Dataset.from_pandas(traindf)
        dev_dataset = Dataset.from_pandas(devdf)
        test_dataset = Dataset.from_pandas(testdf)
        
        my_dataset = DatasetDict({
            "train": train_dataset,
            "validation": dev_dataset,
            "test": test_dataset
        })
        print(my_dataset["train"][0])
    
        #? トークナイズ
        tokenized_dataset = my_dataset.map(self.tokenize, batched=True)
        print(tokenized_dataset)
        
        return tokenized_dataset 
    
    def train(self, tokenized_dataset) -> None:
        #? Train設定値準備
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=5).to(device)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        dataset_size = 30000
        batch_size = 64
        logging_steps = dataset_size // batch_size
        train_args = TrainingArguments(
            output_dir=join(self.RESULTPATH, "touhoku-bert-base--classify"),
            num_train_epochs=20,
            learning_rate=1e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            disable_tqdm=False,
            logging_steps=logging_steps,
            push_to_hub=False,
            log_level="error",
            save_total_limit=5,
            load_best_model_at_end=True,
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            save_strategy='epoch',
            metric_for_best_model="accuracy"
        )
        
        #? 訓練
        trainer = Trainer(
            model = model,
            args=train_args,
            compute_metrics=self.compute_metrics,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train(ignore_keys_for_eval=['last_hidden_state', 'hidden_states', 'attentions'])
        
        #? 推論結果
        trainer.evaluate()
        
        #? モデル保存
        trainer.save_state()
        trainer.save_model()
        
        output = trainer.predict(tokenized_dataset["test"])
        test_preds = np.argmax(output.predictions, axis=1)
        test_preds.astype('int32')
        test_preds = [x-2 for x in test_preds]
        
        print(test_preds)
        
        np.savetxt("./test.txt", test_preds, fmt="%.0f")
        
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
    def chenge_label_range(self, df):
        df['labels'] = df['labels'].replace([-2,-1,0,1,2],[0,1,2,3,4])
        return df  
    
    def create_path(self, pathtext:str) -> str:
        return abspath(pathtext)
    
    def tokenize(self, record:object, ):
        return self.tokenizer(record["text"], truncation=True, padding=True)
    
    def barttest(self):
        checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        
        sequences = ['建設中の建物の利用目的も変更になるだろうなあ。', '演劇とかも同時並行でやっている。演劇の危機にどう思ってるか知りたかった。']
        
        tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
        
        output = model(**tokens)
        
        print(output)

if __name__ == '__main__':
    Main().main()