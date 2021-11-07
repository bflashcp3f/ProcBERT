
import torch

from collections import defaultdict, Counter, OrderedDict
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification, BertConfig, BertTokenizer, BertPreTrainedModel, AdamW
from transformers import RobertaTokenizer, RobertaForTokenClassification

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class BertForNER(BertForTokenClassification):

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
            labels=None, head_tags=None, head_flags=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        #         print("sequence_output.size()", sequence_output.size())
        #         print("labels.size()", labels.size())
        #         print("attention_mask.size()", attention_mask.size())
        #         print("head_flags.size()", head_flags.size())
        #         print("head_flags", head_flags)
        #         print("attention_mask", attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        #         print("logits.size()", logits.size())

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


class RobertaForNER(RobertaForTokenClassification):

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
            labels=None, head_tags=None, head_flags=None):

        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        #         print("sequence_output.size()", sequence_output.size())
        #         print("labels.size()", labels.size())
        #         print("attention_mask.size()", attention_mask.size())
        #         print("head_flags.size()", head_flags.size())
        #         print("head_flags", head_flags)
        #         print("attention_mask", attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        #         print("logits.size()", logits.size())

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