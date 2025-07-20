import torch
from torch import nn
import transformers
from transformers.activations import ACT2FN
if transformers.__version__ == '3.4.0':
    from transformers.modeling_roberta import (
        RobertaEmbeddings,
        RobertaEncoder,
        RobertaPreTrainedModel,
        RobertaPooler,
    )
else:
     the latest version
    from transformers.models.roberta.modeling_roberta import(
        RobertaEmbeddings,
        RobertaEncoder,
        RobertaPreTrainedModel,
        RobertaPooler,
    )

class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_attentions=None, output_hidden_states=None):
        assert input_ids is not None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

class RobertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class RobertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RobertaPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class RobertaPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RobertaLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 4)
    
    def forward(self, sequence_ouutput, pooled_output):
        prediction_scores = self.predictions(sequence_ouutput)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class RobertaForPreTraining(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.cls = RobertaPreTrainingHeads(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, topic=None):
        batch_size = input_ids.shape[0]
        sequence_output, pooled_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        pred_topic = seq_relationship_score.argmax(dim=1)
        tp = torch.sum(pred_topic.view(-1) == topic.view(-1)).item()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        tp_temp = list((pred_topic.view(-1) == sentiment.view(-1)))

        tp0, tp1, tp2, tp3 = 0, 0, 0, 0
        fp0, fp1, fp2, fp3 = 0, 0, 0, 0
        fn0 = fn1 = fn2 = fn3 = 0
        for i in range(batch_size):
            predicted = pred_topic[i].item()
            actual = topic[i].item()
             如果预测 == 实际
            if tp_temp[i] == False:
            print(self.tokenizer.decode(input_ids[i]))
            print(pred_sentiment.view(-1)[i])
            print(sentiment.view(-1)[i])
        
            if predicted == actual:
                if predicted == 0:
                    tp0 += 1
                elif predicted == 1:
                    tp1 += 1
                elif predicted == 2:
                    tp2 += 1
                elif predicted == 3:
                    tp3 += 1
                else:
                    raise Exception("Topic Label Error (predicted out of range)")
            else:
                 FP: 预测为某类别，但实际并不是
                if predicted == 0:
                    fp0 += 1
                elif predicted == 1:
                    fp1 += 1
                elif predicted == 2:
                    fp2 += 1
                elif predicted == 3:
                    fp3 += 1
                else:
                    raise Exception("Topic Label Error (predicted out of range)")

                 FN: 实际是某类别，但预测不是
                if actual == 0:
                    fn0 += 1
                elif actual == 1:
                    fn1 += 1
                elif actual == 2:
                    fn2 += 1
                elif actual == 3:
                    fn3 += 1
                else:
                    raise Exception("Topic Label Error (actual out of range)")
         6. 计算 TP（所有类别合并）
        tp = tp0 + tp1 + tp2 + tp3
        
        loss = loss_fct(seq_relationship_score.view(-1, 4), topic.view(-1))
        return loss, tp, tp0, tp1, tp2, tp3, fp0, fp1, fp2, fp3, fn0, fn1, fn2, fn3
