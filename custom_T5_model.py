import torch.nn as nn
from typing import Optional
import os
from safetensors.torch import save_model
from transformers import Trainer

#Feature Extraction Model
class CustomT5Model(nn.Module):
    def __init__(self, base_model):
        super(CustomT5Model, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(base_model.config.d_model, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 3)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier[-1].out_features), labels.view(-1))
        return (loss, logits) if loss is not None else logits
    
#Save model fnct.
class CustomTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_model(self.model, os.path.join(output_dir, 'classifier.safetensors'))
        self.tokenizer.save_pretrained(output_dir)