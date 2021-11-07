
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification, BertPreTrainedModel, AdamW
from transformers import RobertaConfig, RobertaModel


class BertSimpleEMES(BertPreTrainedModel):

    def __init__(self, config):
        super(BertSimpleEMES, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                subj_ent_start=None, obj_ent_start=None):

        # Get the embedding for paraphrase
        # outputs = self.bert(input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :])
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # outputs = self.bert(input_ids_new, attention_mask=attention_mask_new)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Take the representation for 'SUBJ' token
        subj_ent_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_ent_start)])

        # Take the representation for 'OBJ' token
        obj_ent_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_ent_start)])


        ent_output = torch.cat([subj_ent_output, obj_ent_output], dim=1)

        bag_output = self.dropout(ent_output)

        logits = self.classifier(bag_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        #         print("outputs: ", outputs)

        return outputs  # (loss), logits, (hidden_states), (attentions)


# class RobertaSimpleEMES(RobertaPreTrainedModel):
class RobertaSimpleEMES(BertPreTrainedModel):
    config_class = RobertaConfig
    #     pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaSimpleEMES, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                subj_ent_start=None, obj_ent_start=None):

        # Get the embedding for paraphrase
        # outputs = self.bert(input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :])
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # outputs = self.bert(input_ids_new, attention_mask=attention_mask_new)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Take the representation for 'SUBJ' token
        subj_ent_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_ent_start)])

        # Take the representation for 'OBJ' token
        obj_ent_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_ent_start)])

        ent_output = torch.cat([subj_ent_output, obj_ent_output], dim=1)

        bag_output = self.dropout(ent_output)

        logits = self.classifier(bag_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        #         print("outputs: ", outputs)

        return outputs  # (loss), logits, (hidden_states), (attentions)