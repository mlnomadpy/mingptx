import grain.python as pygrain
from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass

from config import DataConfig, ModelConfig, TrainConfig

@dataclass
class HFTextDataset:
    hf_dataset: load_dataset
    tokenizer: AutoTokenizer
    maxlen: int

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        text = self.hf_dataset[idx]["text"]
        encoding = self.tokenizer.encode(
            text, 
            truncation=True, 
            max_length=self.maxlen, 
            padding=False
        )
        
        if self.tokenizer.pad_token_id is None:
            pad_id = self.tokenizer.eos_token_id
        else:
            pad_id = self.tokenizer.pad_token_id

        padded_encoding = encoding + [pad_id] * (self.maxlen - len(encoding))
        return padded_encoding

def load_text_dataset(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer: AutoTokenizer):
    hf_dataset = load_dataset(d_config.dataset_name, split=d_config.split)
    dataset = HFTextDataset(hf_dataset, tokenizer, m_config.maxlen)

    sampler = pygrain.IndexSampler(
        num_records=len(dataset),
        shuffle=True,
        seed=42,
        shard_options=pygrain.NoSharding(),
        num_epochs=t_config.num_epochs,
    )

    dataloader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=d_config.batch_size, drop_remainder=True)],
    )
    return dataloader 