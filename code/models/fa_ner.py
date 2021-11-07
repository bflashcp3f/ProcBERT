
import torch

from collections import defaultdict, Counter, OrderedDict
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification, BertConfig, BertTokenizer, BertPreTrainedModel, AdamW
from transformers import RobertaModel, RobertaTokenizer, RobertaForTokenClassification

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class RobertaForNERFA(BertPreTrainedModel):

    def __init__(self, config):
        super(RobertaForNERFA, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, head_tags=None, head_flags=None, data_types=None):

        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        sequence_output = torch.cat((sequence_output, sequence_output, sequence_output), 2)

        data_types_new = data_types.unsqueeze(-1).repeat(1, 1, self.hidden_size).view(sequence_output.size()[0],
                                                                                      -1).unsqueeze(1).repeat(1, sequence_output.size()[1], 1)

        assert data_types_new.size() == sequence_output.size()

        sequence_output = torch.mul(sequence_output, data_types_new)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # print("logits.size()", logits.size())

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:

                if head_tags is None:
                    active_loss = attention_mask.view(-1) == 1
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                #                 print("active_labels.size()", active_labels.size())
                else:
                    active_loss = head_flags.view(-1) == 1
                    active_labels = torch.where(
                        active_loss, head_tags.view(-1), torch.tensor(loss_fct.ignore_index).type_as(head_tags)
                    )

                active_logits = logits.view(-1, self.num_labels)
                #                 print("active_logits.size()", active_logits.size())

                loss = loss_fct(active_logits, active_labels)
            #                 print("loss.size()", loss.size())
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)



class BertForNERFA(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForNERFA, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # self.bert_src = BertModel(config)
        self.bert = BertModel(config)
        # self.bert_gen = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels)

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
            labels=None, head_tags=None, head_flags=None, data_types=None):

        # outputs_src = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs_tgt = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs_gen = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #
        # sequence_output_src = outputs_src[0]
        # sequence_output_tgt = outputs_tgt[0]
        # sequence_output_gen = outputs_gen[0]

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        # sequence_output = torch.cat((sequence_output_src, sequence_output_tgt, sequence_output_gen), 2)
        # sequence_output = torch.cat((sequence_output_tgt, sequence_output_tgt, sequence_output_tgt), 2)
        sequence_output = torch.cat((sequence_output, sequence_output, sequence_output), 2)

        # print("sequence_output.size()", sequence_output.size())
        # print("labels.size()", labels.size())
        # print("attention_mask.size()", attention_mask.size())
        # print("head_flags.size()", head_flags.size())
        # print("data_types.size()", data_types.size())
        # print("head_flags", head_flags)
        # print("attention_mask", attention_mask)
        # print("data_types", data_types)

        # data_types_new = data_types.unsqueeze(-1).repeat(1, 1, 2).view(sequence_output.size()[0], -1).unsqueeze(1).repeat(1, 2, 1)
        data_types_new = data_types.unsqueeze(-1).repeat(1, 1, self.hidden_size).view(sequence_output.size()[0], -1).unsqueeze(1).repeat(1, sequence_output.size()[1], 1)
        # print("data_types_new.size()", data_types_new.size())
        # print("data_types_new", data_types_new)
        assert data_types_new.size() == sequence_output.size()

        sequence_output = torch.mul(sequence_output, data_types_new)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # print("logits.size()", logits.size())

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:

                if head_tags is None:
                    active_loss = attention_mask.view(-1) == 1
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                #                 print("active_labels.size()", active_labels.size())
                else:
                    active_loss = head_flags.view(-1) == 1
                    active_labels = torch.where(
                        active_loss, head_tags.view(-1), torch.tensor(loss_fct.ignore_index).type_as(head_tags)
                    )

                active_logits = logits.view(-1, self.num_labels)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)



class BertForNERFAMultiEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForNERFAMultiEncoder, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # self.bert_src = BertModel(config)
        self.bert = BertModel(config)
        self.bert_src = self.bert
        self.bert_tgt = BertModel(config)
        self.bert_gen = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels)

        self.init_weights()

    def init_multi_encoder(self):
        self.bert_tgt.load_state_dict(self.bert.state_dict())
        self.bert_gen.load_state_dict(self.bert.state_dict())

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
            labels=None, head_tags=None, head_flags=None, data_types=None):

        outputs_src = self.bert_src(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs_tgt = self.bert_tgt(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs_gen = self.bert_gen(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output_src = outputs_src[0]
        sequence_output_tgt = outputs_tgt[0]
        sequence_output_gen = outputs_gen[0]

        sequence_output = torch.cat((sequence_output_src, sequence_output_tgt, sequence_output_gen), 2)

        # outputs = self.bert_tgt(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # sequence_output = outputs[0]
        # sequence_output = torch.cat((sequence_output, sequence_output, sequence_output), 2)

        # print("sequence_output.size()", sequence_output.size())
        # print("labels.size()", labels.size())
        # print("attention_mask.size()", attention_mask.size())
        # print("head_flags.size()", head_flags.size())
        # print("data_types.size()", data_types.size())
        # print("head_flags", head_flags)
        # print("attention_mask", attention_mask)
        # print("data_types", data_types)

        # data_types_new = data_types.unsqueeze(-1).repeat(1, 1, 2).view(sequence_output.size()[0], -1).unsqueeze(1).repeat(1, 2, 1)
        data_types_new = data_types.unsqueeze(-1).repeat(1, 1, self.hidden_size).view(sequence_output.size()[0], -1).unsqueeze(1).repeat(1, sequence_output.size()[1], 1)
        # print("data_types_new.size()", data_types_new.size())
        # print("data_types_new", data_types_new)
        assert data_types_new.size() == sequence_output.size()

        sequence_output = torch.mul(sequence_output, data_types_new)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # print("logits.size()", logits.size())

        outputs = (logits,) + outputs_tgt[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:

                if head_tags is None:
                    active_loss = attention_mask.view(-1) == 1
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                #                 print("active_labels.size()", active_labels.size())
                else:
                    active_loss = head_flags.view(-1) == 1
                    active_labels = torch.where(
                        active_loss, head_tags.view(-1), torch.tensor(loss_fct.ignore_index).type_as(head_tags)
                    )

                active_logits = logits.view(-1, self.num_labels)
                #                 print("active_logits.size()", active_logits.size())

                loss = loss_fct(active_logits, active_labels)
            #                 print("loss.size()", loss.size())
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

