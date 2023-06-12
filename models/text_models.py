from transformers import BertForSequenceClassification
import types
import torch


class BertWrapper(torch.nn.Module):
    # adapted from https://github.com/facebookresearch/BalancingGroups/blob/main/models.py
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(input_ids=x[:, :, 0], attention_mask=x[:, :, 1],
                          token_type_ids=x[:, :, 2]).logits


def _bert_replace_fc(model):
    model.fc = model.classifier
    delattr(model, "classifier")

    def classifier(self, x):
        return self.fc(x)

    model.classifier = types.MethodType(classifier, model)

    model.base_forward = model.forward

    def forward(self, x):
        return self.base_forward(input_ids=x[:, :, 0], attention_mask=x[:, :, 1],
                                 token_type_ids=x[:, :, 2]).logits

    model.forward = types.MethodType(forward, model)
    return model


def bert_pretrained(output_dim):
    return _bert_replace_fc(
        BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=output_dim))
