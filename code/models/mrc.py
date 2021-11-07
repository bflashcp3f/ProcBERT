

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertPreTrainedModel, AdamW, BertConfig, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification

from keras.preprocessing.sequence import pad_sequences
from torch.nn import CrossEntropyLoss, MSELoss



class BertMRC(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super(BertMRC, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                cls_pos=None):

        # print("input_ids.size(): ", input_ids.size())
        # print("attention_mask.size(): ", attention_mask.size())
        # print("labels.size(): ", labels.size())
        # print("cls_pos.size(): ", cls_pos.size())

        # Get the embedding for paraphrase
        # outputs = self.bert(input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :])
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # outputs = self.bert(input_ids_new, attention_mask=attention_mask_new)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # print("sequence_output.size(): ", sequence_output.size())
        # print("pooled_output.size(): ", pooled_output.size())
        #
        # print("sequence_output[0][5]: ", sequence_output[0][5])
        # print("sequence_output[0][23]: ", sequence_output[0][23])
        # print("sequence_output[0][31]: ", sequence_output[0][31])
        # print("cls_pos: ", cls_pos)

        # Take select the representation from coresponding index
        cls_output = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(sequence_output, cls_pos)])
        # print("cls_output.size()", cls_output.size())
        # print("cls_output: ", cls_output)

        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # print("logits.size(): ", logits.size())
        # print("labels.size(): ", labels.size())
        #
        # print("logits: ", logits)
        # print("labels: ", labels)
        #
        # print("logits.view(-1, self.num_labels): ", logits.view(-1, self.num_labels))
        # print("labels.view(-1, 1): ", labels.view(-1, 1))
        # print("labels.view(-1): ", labels.view(-1))

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

            # print("outputs: ", outputs)

        return outputs  # (loss), logits, (hidden_states), (attentions)


# class RobertaMRC(RobertaPreTrainedModel):
class RobertaMRC(BertPreTrainedModel):
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaMRC, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                cls_pos=None):

        # print("input_ids.size(): ", input_ids.size())
        # print("attention_mask.size(): ", attention_mask.size())
        # print("labels.size(): ", labels.size())
        # print("cls_pos.size(): ", cls_pos.size())

        # Get the embedding for paraphrase
        # outputs = self.bert(input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :])
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # outputs = self.bert(input_ids_new, attention_mask=attention_mask_new)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # print("sequence_output.size(): ", sequence_output.size())
        # print("pooled_output.size(): ", pooled_output.size())
        #
        # print("sequence_output[0][5]: ", sequence_output[0][5])
        # print("sequence_output[0][23]: ", sequence_output[0][23])
        # print("sequence_output[0][31]: ", sequence_output[0][31])
        # print("cls_pos: ", cls_pos)

        # Take select the representation from coresponding index
        cls_output = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(sequence_output, cls_pos)])
        # print("cls_output.size()", cls_output.size())
        # print("cls_output: ", cls_output)

        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # print("logits.size(): ", logits.size())
        # print("labels.size(): ", labels.size())
        #
        # print("logits: ", logits)
        # print("labels: ", labels)
        #
        # print("logits.view(-1, self.num_labels): ", logits.view(-1, self.num_labels))
        # print("labels.view(-1, 1): ", labels.view(-1, 1))
        # print("labels.view(-1): ", labels.view(-1))

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

            # print("outputs: ", outputs)

        return outputs  # (loss), logits, (hidden_states), (attentions)
