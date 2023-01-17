import torch
import math
import numpy as np
import os

import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f"runs/alpha_model_mean")

class MLMTrainer(object):
    
    FOLDER = "mlm_trainings"
    
    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        tokenizer,
        mask_probability: float = 0.15,
        random_flip_mask_probability: float = 0.15,
        device: str = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._tokenizer = tokenizer
        self._iterations = 0
        self._name = name
        
        self._random_flip_mask_probability = random_flip_mask_probability
        self._mask_probability = mask_probability
        
        self._device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self._folder = os.path.join(self.FOLDER, self._name)
        
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
            
        self._pad_id = tokenizer(tokenizer.pad_token)["input_ids"][1]
        self._mask_id = tokenizer(tokenizer.mask_token)["input_ids"][1]
        self._token_max_id = tokenizer.vocab_size

        self._writer_train = SummaryWriter(f"runs/{self._name}_train")
        self._writer_val = SummaryWriter(f"runs/{self._name}_val")
            
    def _generate_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        attention_mask = attention_mask.to(self._device)
        
        mask_mask, permuted_tokens, labels = mask_objective(
            input_ids=input_ids.to(self._device),
            attention_mask=attention_mask.to(self._device),
            pad_id=self._pad_id,
            mask_id=self._mask_id,
            token_max_id=self._token_max_id,
            mask_probability=self._mask_probability,
            random_flip_mask_probability=self._random_flip_mask_probability,
        )
        
        return mask_mask, permuted_tokens, labels, attention_mask
            
    def _train_iteration(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        
        mask, input_ids, labels, attention_mask = self._generate_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        self._model.zero_grad()
        
        output = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        l = mlm_loss(
            logits=output.logits,
            labels=labels,
            mask=mask,
            token_max_id=self._token_max_id,
        )

        l.backward()
        
        self._optimizer.step()
        
        self._writer_train.add_scalar(f"loss", l.item(), self._iterations)
        
    def _val_iteration(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        
        mask, input_ids, labels, attention_mask = self._generate_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        output = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        l = mlm_loss(
            logits=output.logits,
            labels=labels,
            mask=mask,
            token_max_id=self._token_max_id,
        )
        
        self._writer_val.add_scalar(f"loss", l.item(), self._iterations)

    def run(
        self,
        dataloader_train,
        dataloader_val,
        epochs: int = 100,
        validation_iterations: int = 1000,
        save_iterations: int = 10000,
    ):
        
        iterator_val = iter(dataloader_val)
        validation_batch = next(iterator_val)
        
        for epoch in range(epochs):
        
            iterator_train = iter(dataloader_train)
            
            while True:

                try:
                    batch = next(iterator_train)
                except StopIteration:
                    break 
                
                self._train_iteration(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            
                self._iterations += 1
                
                if self._iterations % validation_iterations == 0:
                    
                    try:
                        validation_batch = next(iterator_val)
                    except StopIteration:
                        iterator_val = iter(dataloader_val) 
                        validation_batch = next(iterator_val)

                    self._val_iteration(
                        input_ids=validation_batch["input_ids"],
                        attention_mask=validation_batch["attention_mask"],
                    )
    
            if self._iterations % save_iterations == 0:
                
                print("saving model")
                
                torch.save(
                    self._model.state_dict(),
                    os.path.join(self._folder, f"{self._iterations}.ckp")
                )

def mask_objective(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
    mask_id: int,
    token_max_id: int,
    mask_probability: float = 0.15,
    random_flip_mask_probability: float = 0.15,
):
    
    device = input_ids.device

    mask_mask = (
        (torch.rand(input_ids.shape).to(device) < mask_probability) & attention_mask
    ).bool()
    
    flip_mask = (
        (torch.rand(input_ids.shape).to(device) < random_flip_mask_probability) & mask_mask
    ).bool()
    
    true_tokens = input_ids.clone()
    
    permuted_tokens = input_ids.clone()
    permuted_tokens[mask_mask] = mask_id
    
    random_tokens = torch.randint(
        low=0,
        high=token_max_id,
        size=permuted_tokens.shape
    ).to(device)
    
    permuted_tokens[flip_mask] = random_tokens[flip_mask]
    
    return mask_mask, permuted_tokens, true_tokens

def mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    token_max_id: int,
):
    
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    
    logits = logits.reshape(-1, token_max_id)
    
    l = loss_func(logits, labels.reshape(-1))
    
    l = l[mask.reshape(-1)]
    
    return l.mean()

class PositionalEncoding(torch.nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_seq_len: int = 512):

        super().__init__()

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))

        pos_enc = torch.zeros(1, max_seq_len, hidden_dim)

        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)

        # register_buffer, not part of state dict, is not going to be exposed to opti,
        # but be part of transfer to device etc..
        self.register_buffer('pos_enc', pos_enc)
        
        self._embedder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        

    def forward(self, batch_size: int, sequence_length: int):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        
        pos_encoded = self.pos_enc[:, :sequence_length, :].repeat(batch_size, 1, 1)
        
        pos_embedded = self._embedder(pos_encoded)
        
        return pos_embedded

class ProgressionEmbeddings(torch.nn.Module):

    def __init__(self, hidden_dim: int, input_dim: int = 1, dropout: float = 0.1):

        super().__init__()

        self._embedder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor):

        y = self._embedder(x)
      
        return y


class TransformerModel(torch.nn.Module):

    def __init__(
        self,
        nlayers_tokens: int,
        nlayers_batches: int,
        input_feature_dim: int = 768,
        progression_dim: int = 32,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        dim_feedforward: int = 2048,
    ):

        super().__init__()
        
        hidden_dim = progression_dim + input_feature_dim

        self._pos_encoder = PositionalEncoding(
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        
        self._progression_embedder = ProgressionEmbeddings(
            input_dim=1,
            hidden_dim=progression_dim,
            dropout=dropout,
        )
        
        self._token_decoder_layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True
                )
                for _ in range(nlayers_tokens)
            ]
        )
        
        
        self._batch_encoder_layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True
                )
                for _ in range(nlayers_batches)
            ]
        )

        self._alpha_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        

    def forward(
        self,
        x_input: torch.Tensor,
        x_output: torch.Tensor,
        x_progression: torch.Tensor,
    ):
        
        input_org_shape = x_input.shape
        
        progression_embedded = self._progression_embedder(
            x_progression
        ).unsqueeze(1).unsqueeze(1).repeat(
            1, input_org_shape[1], input_org_shape[2], 1
        )
        
        x_input = torch.cat(
            tensors=(
                x_input,
                progression_embedded
            ),
            dim=-1
        )
        
        x_output = torch.cat(
            tensors=(
                x_output,
                progression_embedded
            ),
            dim=-1
        )
        
        input_shape = x_input.shape
        b, bb, s, f = input_shape
        
        ## Put all tthins in batch dim 
        
        x_input = x_input.reshape(-1, s, f)
        x_output = x_output.reshape(-1, s, f)
        
        ##
        
        pos_encoded = self._pos_encoder(
            batch_size=x_input.shape[0],
            sequence_length=x_input.shape[1],
        )

        ## Add the pos encoded stuff
        x_input = x_input + pos_encoded
        x_output = x_output + pos_encoded
        
        y_decoder = x_output
        memory = x_input
        
        for layer in self._token_decoder_layers:
            y_decoder = layer(y_decoder, memory)
          
        # Do pooling, take the first token
        y_decoder_pooled = y_decoder.mean(dim=1)
        
        # rehsape back to the org split betweeen layer and token batches
        y_decoder_pooled = y_decoder_pooled.reshape(input_org_shape[0], input_org_shape[1], -1)
        
        y_encoder = y_decoder_pooled
        
        for layer in self._batch_encoder_layers:
            y_encoder = layer(y_encoder)
        
        # Do pooling, take mean over the seq-dim
        y_encoder = y_encoder.mean(dim=1)
        
        y_alphas = self._alpha_layers(y_encoder)
        
        return y_alphas

    def save(self, folder: str, name: str):

        if not os.path.exists(folder):
            os.makedirs(folder)

        path = os.path.join(folder, name)

        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)

        return path

    def load(self, folder: str, name: str):

        path = os.path.join(folder, name)
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])


from typing import Tuple
import copy

def permute_layer(layer_true: torch.nn.Module, layer_permute: torch.nn.Module, alpha: float):
    
    return layer_true * alpha  + layer_permute * (1 - alpha)

def permute_layers(layers: torch.nn.Module, alpha: float, sigma_factor: float = 1):
    
    layers_permuted = copy.deepcopy(layers)
    
    iterable = zip(
        layers.named_parameters(),
        layers_permuted.named_parameters(),
    )    
    
    for (name_true, param_true), (name_permuted, param_permuted) in iterable:
        
        if "bias" == name_true:
            
            mean, std = 0, 0

        elif "layernorm" == name_true.lower():

            mean, std = 1, 0
            
        else:
            pass 
        
        mean, std = torch.mean(param_true).item(), torch.std(param_true).item() * sigma_factor
            
        # print(name_true, torch.mean(param_true).item())
        
        torch.nn.init.normal_(
            param_permuted,
            mean,
            std * alpha
        )

        param_permuted_updated = permute_layer(
            layer_true=param_true,
            layer_permute=param_permuted,
            alpha=0
        )
        
        with torch.no_grad():
            param_permuted.copy_(param_permuted_updated)
        
    return layers_permuted

def create_labels(
    layers: torch.nn.ModuleList,
    batch_size: int,
    hidden_states: Tuple,
    sigma_factor: float = 1,
    alpha_min: float = 0,
    alpha_max: float = 1,
):
    
    layer_indicies = np.random.randint(1, len(layers), batch_size)

    hidden_state_inputs = []
    hidden_state_permuted_outputs = []
    alphas = []

    for layer_index in layer_indicies:

        alpha = np.random.uniform(alpha_min, alpha_max)

        layer_permuted = permute_layers(
            layers[layer_index],
            alpha=alpha,
            sigma_factor=sigma_factor
        )

        hidden_state_input = hidden_states[layer_index]

        with torch.no_grad():
            hidden_state_permuted_output = layer_permuted(hidden_state_input)[0]

        hidden_state_inputs.append(hidden_state_input)
        hidden_state_permuted_outputs.append(hidden_state_permuted_output)
        alphas.append(alpha)

    hidden_state_inputs = torch.stack(hidden_state_inputs)
    hidden_state_permuted_outputs = torch.stack(hidden_state_permuted_outputs)
    alphas = torch.Tensor(alphas).unsqueeze(1)
    
    layer_indicies_progression = (
        torch.Tensor(layer_indicies) / len(layers)
    ).unsqueeze(1)
    
    return (
        hidden_state_inputs,
        hidden_state_permuted_outputs,
        alphas,
        layer_indicies_progression,
    )