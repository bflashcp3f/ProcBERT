
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification, BertPreTrainedModel, AdamW
from transformers import RobertaConfig, RobertaModel

class RobertaSimpleEMESFA(BertPreTrainedModel):
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaSimpleEMESFA, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        # self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 6, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                subj_ent_start=None, obj_ent_start=None, data_types=None):

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
        # print("subj_ent_output.size()", subj_ent_output.size())
        # print("obj_ent_output.size()", obj_ent_output.size())
        # print("ent_output.size()", ent_output.size())

        ent_output = torch.cat((ent_output, ent_output, ent_output), 1)
        # print("ent_output.size()", ent_output.size())

        # print("data_types", data_types)
        # print("data_types.size()", data_types.size())
        data_types_new = data_types.unsqueeze(-1).repeat(1, 1, self.hidden_size*2).view(ent_output.size()[0], -1)
        # data_types_new = data_types.unsqueeze(-1).repeat(1, 1, self.hidden_size).view(sequence_output.size()[0], -1)
        # print("data_types_new", data_types_new)
        # print("data_types_new.size()", data_types_new.size())
        assert data_types_new.size() == ent_output.size()

        ent_output = torch.mul(ent_output, data_types_new)

        bag_output = self.dropout(ent_output)

        # print(bag_output.dtype)
        # print(bag_output.half().dtype)
        # print(self.classifier.dtype)
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

class BertSimpleEMESFA(BertPreTrainedModel):

    def __init__(self, config):
        super(BertSimpleEMESFA, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        # self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 6, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                subj_ent_start=None, obj_ent_start=None, data_types=None):

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
        # print("subj_ent_output.size()", subj_ent_output.size())
        # print("obj_ent_output.size()", obj_ent_output.size())
        # print("ent_output.size()", ent_output.size())

        ent_output = torch.cat((ent_output, ent_output, ent_output), 1)
        # print("ent_output.size()", ent_output.size())

        # print("data_types", data_types)
        # print("data_types.size()", data_types.size())
        data_types_new = data_types.unsqueeze(-1).repeat(1, 1, self.hidden_size*2).view(ent_output.size()[0], -1)
        # data_types_new = data_types.unsqueeze(-1).repeat(1, 1, self.hidden_size).view(sequence_output.size()[0], -1)
        # print("data_types_new", data_types_new)
        # print("data_types_new.size()", data_types_new.size())
        assert data_types_new.size() == ent_output.size()

        ent_output = torch.mul(ent_output, data_types_new)

        bag_output = self.dropout(ent_output)

        # print(bag_output.dtype)
        # print(bag_output.half().dtype)
        # print(self.classifier.dtype)
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

class BertSimpleEMESFAMultiEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertSimpleEMESFAMultiEncoder, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.bert_src = self.bert
        self.bert_tgt = BertModel(config)
        self.bert_gen = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        # self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 6, self.config.num_labels)

        self.init_weights()

    def init_multi_encoder(self):
        # self.bert_tgt.load_state_dict(self.bert.state_dict())
        # print("Compare bert_tgt and bert_src.")
        # self.compare_models(self.bert_tgt, self.bert_src)
        #
        # self.bert_gen.load_state_dict(self.bert.state_dict())
        # print("Compare bert_gen and bert_src.")
        # self.compare_models(self.bert_gen, self.bert_src)

        pass

    def compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                subj_ent_start=None, obj_ent_start=None, data_types=None):

        # Get the embedding for paraphrase
        # outputs = self.bert(input_ids, attention_mask=attention_mask)
        # sequence_output = outputs[0]
        # pooled_output = outputs[1]

        outputs_src = self.bert_src(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs_tgt = self.bert_tgt(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs_gen = self.bert_gen(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output_src = outputs_src[0]
        # sequence_output_tgt = outputs_tgt[0]
        # sequence_output_gen = outputs_gen[0]

        subj_ent_output_src = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output_src, subj_ent_start)])
        obj_ent_output_src = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output_src, obj_ent_start)])
        ent_output_src = torch.cat([subj_ent_output_src, obj_ent_output_src], dim=1)

        # subj_ent_output_tgt = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output_tgt, subj_ent_start)])
        # obj_ent_output_tgt = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output_tgt, obj_ent_start)])
        # ent_output_tgt = torch.cat([subj_ent_output_tgt, obj_ent_output_tgt], dim=1)

        # subj_ent_output_gen = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output_gen, subj_ent_start)])
        # obj_ent_output_gen = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output_gen, obj_ent_start)])
        # ent_output_gen = torch.cat([subj_ent_output_gen, obj_ent_output_gen], dim=1)

        # ent_output = torch.cat((ent_output_src, ent_output_tgt, ent_output_gen), 1)
        # ent_output = torch.cat((ent_output_tgt, ent_output_src, ent_output_gen), 1)
        # ent_output = torch.cat((ent_output_gen, ent_output_gen, ent_output_gen), 1)
        # ent_output = torch.cat((ent_output_src, ent_output_tgt, ent_output_tgt), 1)
        ent_output = torch.cat((ent_output_src, ent_output_src, ent_output_src), 1)
        # ent_output = torch.cat((ent_output_tgt, ent_output_tgt, ent_output_tgt), 1)
        # ent_output = torch.cat((ent_output_tgt, ent_output_src, ent_output_src), 1)
        # ent_output = torch.cat((ent_output_tgt, ent_output_tgt, ent_output_src), 1)
        # print("ent_output.size()", ent_output.size())

        # print("data_types", data_types)
        # print("data_types.size()", data_types.size())
        data_types_new = data_types.unsqueeze(-1).repeat(1, 1, self.hidden_size*2).view(ent_output.size()[0], -1)
        # data_types_new = data_types.unsqueeze(-1).repeat(1, 1, self.hidden_size).view(sequence_output.size()[0], -1)
        # print("data_types_new", data_types_new.detach().cpu().numpy().tolist())
        # print("data_types_new.size()", data_types_new.size())
        assert data_types_new.size() == ent_output.size()

        ent_output = torch.mul(ent_output, data_types_new)

        # bag_output = self.dropout(ent_output_src)
        # bag_output = self.dropout(ent_output_tgt)
        bag_output = self.dropout(ent_output)

        logits = self.classifier(bag_output)
        # logits = self.classifier(bag_output.half())

        outputs = (logits,) + outputs_src[2:]  # add hidden states and attention if they are here

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


