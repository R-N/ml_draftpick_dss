import os
import time

import numpy as np
from tqdm import tqdm

from ..utils import *
from ..NeuralNet import NeuralNet
from ...predicting.modules import GlobalPooling1D
from ...predicting.embedding import scaled_sqrt_factory, create_embedding_sizes
from ...predicting.encoding import HeroLabelEncoder, HeroOneHotEncoder

import torch
import torch.optim as optim

from .model import DraftingAgentModel

def split_pis(pis, n=120):
    return pis[..., :n], pis[..., n:]

def to_tensor(b):
    return torch.FloatTensor(np.array(b).astype(np.float64))

def to_gpu(b):
    return b.contiguous().cuda()

class DraftingNeuralNet(NeuralNet):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game, *args, **kwargs):
        self.model = DraftingAgentModel(*args, **kwargs)

    def prepare_training(
        self,
        epochs=10,
        lr=1e-3,
        dropout=0.1,
        batch_size=64,
    ):
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.batch_size = batch_size
        self.cuda = torch.cuda.is_available()
        


    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        self.model.train()
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                boards = [
                    to_tensor(b) if i < 5 else b
                    for i, b in enumerate(zip(boards)) 
                ]

                tensors = boards[:4]
                legal_mask = boards[4]
                count = to_tensor(list(zip(boards[-3]))[-1])
                next_count = to_tensor(list(zip(boards[-2]))[-1])

                pis_1, pis_2 = split_pis(pis)
                target_pis_1 = to_tensor(pis_1)
                target_pis_2 = to_tensor(pis_2)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.cuda:
                    tensors = [to_gpu(t) for t in tensors]
                    legal_mask = legal_mask.contiguous().cuda()
                    count, next_count = [to_gpu(t) for t in (count, next_count)]
                    target_pis, target_pis_2, target_vs = [to_gpu(t) for t in (target_pis, target_pis_2, target_vs)]

                # compute output
                out_pi, out_v = self.model(*tensors, count=count, next_count=next_count, legal_mask=legal_mask)
                out_pi_1, out_pi_2 = split_pis(out_pi)
                l_pi_1 = self.loss_pi(target_pis_1, out_pi_1)
                l_pi_2 = self.loss_pi(target_pis_2, out_pi_2)
                l_pi = l_pi_1 + l_pi_2
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        # timing
        start = time.time()

        # preparing input

        boards = [
            to_tensor(b) if i < 5 else b
            for i, b in enumerate(zip(boards)) 
        ]

        tensors = boards[:4]
        legal_mask = boards[4]
        count = to_tensor(list(zip(boards[-3]))[-1])
        next_count = to_tensor(list(zip(boards[-2]))[-1])
        # predict
        if self.cuda:
            tensors = [to_gpu(t) for t in tensors]
            legal_mask = legal_mask.contiguous().cuda()
            count, next_count = [to_gpu(t) for t in (count, next_count)]
            target_pis, target_pis_2, target_vs = [to_gpu(t) for t in (target_pis, target_pis_2, target_vs)]

        self.model.eval()
        with torch.no_grad():
            pi, v = self.model(*tensors, count=count, next_count=next_count, legal_mask=legal_mask)
            pi_1, pi_2 = split_pis(pi)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


def create_model(
    game=None,
    encoder=7,
    s_embed=4,
    d_hid_encoder=32,
    n_layers_encoder=2,
    activation_encoder=torch.nn.ELU,
    bias_encoder=False,
    n_heads_tf=4,
    d_hid_tf=8,
    n_layers_tf=2,
    n_heads_tf_ban=4,
    d_hid_tf_ban=8,
    n_layers_tf_ban=2,
    activation_tf=torch.nn.ELU,
    d_hid_final=32,
    n_layers_final=3,
    d_hid_final_2=32,
    n_layers_final_2=3,
    activation_final=torch.nn.ELU,
    activation_final_head=torch.nn.Sigmoid,
    bias_final=True,
    n_layers_head=2,
    n_heads=3,
    dropout=0.1,
    pos_encoder=False,
    bidirectional="diff_right",
    pooling=GlobalPooling1D(),
    final_2_mode="double",
    v_pooling="mean",
    agent=DraftingNeuralNet,
    **kwargs,
):
    if isinstance(encoder, int):
        sizes = encoder
    elif isinstance(encoder, HeroLabelEncoder):
        sizes = create_embedding_sizes(
            encoder.x.columns[1:], 
            f=scaled_sqrt_factory(s_embed)
        )
    elif isinstance(encoder, torch.nn.Module):
        sizes = encoder
    elif hasattr(encoder, "dim"):
        sizes = encoder.dim
    else:
        raise ValueError(f"Unknown encoder type: {type(encoder)}")
    _agent = agent(
        game,
        sizes, 
        tf_kwargs={
            "encoder_kwargs":{
                "d_hid": d_hid_encoder,
                "n_layers": n_layers_encoder,
                "activation": activation_encoder,
                "bias": bias_encoder,
                "dropout": dropout,
            }, 
            "tf_encoder_kwargs":{
                "n_heads": n_heads_tf,
                "d_hid": d_hid_tf,
                "n_layers": n_layers_tf,
                "activation": activation_tf,
                "dropout": dropout,
            }, 
            "tf_decoder_kwargs":{
                "n_heads": n_heads_tf,
                "d_hid": d_hid_tf,
                "n_layers": n_layers_tf,
                "activation": activation_tf,
                "dropout": dropout,
            },
        },
        tf_ban_kwargs={
            "encoder_kwargs":{
                "d_hid": d_hid_encoder,
                "n_layers": n_layers_encoder,
                "activation": activation_encoder,
                "bias": bias_encoder,
                "dropout": dropout,
            }, 
            "tf_encoder_kwargs":{
                "n_heads": n_heads_tf_ban,
                "d_hid": d_hid_tf_ban,
                "n_layers": n_layers_tf_ban,
                "activation": activation_tf,
                "dropout": dropout,
            }, 
            "tf_decoder_kwargs":{
                "n_heads": n_heads_tf_ban,
                "d_hid": d_hid_tf_ban,
                "n_layers": n_layers_tf_ban,
                "activation": activation_tf,
                "dropout": dropout,
            },
        },
        final_kwargs={
            "d_hid": d_hid_final,
            "n_layers": n_layers_final,
            "activation": activation_final,
            "bias": bias_final,
            "dropout": dropout,
        },
        final_2_kwargs={
            "d_hid": d_hid_final_2,
            "n_layers": n_layers_final_2,
            "activation": activation_final,
            "bias": bias_final,
            "dropout": dropout,
        },
        head_kwargs={
            "n_heads": n_heads,
            "d_hid": d_hid_final,
            "n_layers": n_layers_head,
            "activation": activation_final,
            "activation_final": activation_final_head,
            "bias": bias_final,
            "dropout": dropout,
        },
        pos_encoder=pos_encoder,
        bidirectional=bidirectional,
        pooling=pooling,
        final_2_mode=final_2_mode,
        v_pooling=v_pooling,
        **kwargs
    )
    return _agent