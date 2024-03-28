import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import ShortTermEncoder, GNN
import copy

class MBIMN(SequentialRecommender):

    def __init__(self, config, dataset):
        super(MBIMN, self).__init__(config, dataset)
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.step = config['num_step']
        self.device = config['device']
        self.mask_ratio = config['mask_ratio']
        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']
        self.enable_ms = config['enable_ms']
        self.dataset = config['dataset']
        self.buy_type = dataset.field2token_id["item_type_list"]['0']
        self.tc = config['temperature_coeffcient']
        self.sub_seq_length = config['MAX_SUB_SEQ_LENGTH']
        self.omega = config['scales']

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.context_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
        self.trm_encoder = ShortTermEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps
            )
        self.effect_weights = nn.Parameter(torch.FloatTensor(config['effect_weight']))
        gnn = GNN(self.hidden_size, self.step)
        self.gnns = nn.ModuleList([copy.deepcopy(gnn) for _ in self.effect_weights])
        self.gnn_linear = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)
        try:
            assert self.loss_type in ['BPR', 'CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        # multi-scale mask
        if self.enable_ms:
            data_mask = (item_seq > 0).unsqueeze(1).unsqueeze(2)
            ms_mask = torch.tensor(np.array(
                [
                    np.tri(item_seq.shape[1], item_seq.shape[1], omega - 1)
                    != np.tri(item_seq.shape[1], item_seq.shape[1], -omega)
                    if type(omega) == int
                    else np.tri(item_seq.shape[1], item_seq.shape[1], int(item_seq.shape[1] * omega) - 1)
                         != np.tri(item_seq.shape[1], item_seq.shape[1], -int(item_seq.shape[1] * omega))
                    for omega in self.omega
                ]), device=item_seq.device
            )
            ms_mask = torch.repeat_interleave(ms_mask.unsqueeze(0), item_seq.shape[0], dim=0)
            attention_mask = (ms_mask.long() + data_mask.long()) > 1

            # bidirectional mask
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

            return attention_mask
        else:
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            return extended_attention_mask

    def _padding_sequence(self, sequence, max_length):

        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def reconstruct_train_data(self, item_seq, type_seq, last_buy):

        """
        Mask item sequence for training.
        """

        last_buy = last_buy.tolist()
        device = item_seq.device
        batch_size = item_seq.size(0)

        zero_padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)
        item_seq = torch.cat((item_seq, zero_padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        type_seq = torch.cat((type_seq, zero_padding.unsqueeze(-1)), dim=-1)
        n_objs = (torch.count_nonzero(item_seq, dim=1)+1).tolist()
        for batch_id in range(batch_size):
            n_obj = n_objs[batch_id]
            item_seq[batch_id][n_obj-1] = last_buy[batch_id]
            type_seq[batch_id][n_obj-1] = self.buy_type

        sequence_instances = item_seq.cpu().numpy().tolist()
        type_instances = type_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        masked_index = []

        for instance_idx, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if index_id == n_objs[instance_idx]-1:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    # type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    # type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)

            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        type_instances = torch.tensor(type_instances, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence, pos_items, masked_index, type_instances

    def reconstruct_test_data(self, item_seq, item_seq_len, item_type):

        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        item_type = torch.cat((item_type, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
            item_type[batch_id][last_position]=self.buy_type
        return item_seq, item_type

    def forward(self, item_seq, type_seq):

        # --Short-term Interest Modeling
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        context_embedding = self.context_embedding(type_seq)
        item_emb = self.item_embedding(item_seq)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        context_embedding = context_embedding + position_embedding
        context_embedding = self.LayerNorm(context_embedding)
        context_embedding = self.dropout(context_embedding)

        scale_encoder_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, context_embedding, scale_encoder_mask, output_all_encoded_layers=True)
        trm_output = trm_output[-1]

        # --Long-term Interest Modeling
        all_sub_sequence = []
        all_sub_nonz = []

        for i in range(self.effect_weights.shape[0]):
            num_elements_per_row = (type_seq == i+1).sum(dim=1)
            sub_seq = torch.split(item_seq[type_seq == i+1], num_elements_per_row.tolist())
            sub_seq_padded = []
            for row in sub_seq:
                if len(row) < self.sub_seq_length:
                    row = torch.cat([row, torch.zeros(self.sub_seq_length - len(row), dtype=row.dtype, device=self.device)])
                else:
                    row = row[-self.sub_seq_length:]
                sub_seq_padded.append(row)
            sub_seq = torch.stack(sub_seq_padded)
            sub_seq_nonz = torch.count_nonzero(sub_seq, dim=1)
            all_sub_nonz.append(copy.deepcopy(sub_seq_nonz))
            sub_seq_nonz[torch.where(sub_seq_nonz == 0)] = 1
            alias_inputs, A, items, mask = self._get_slice(sub_seq)
            hidden = self.item_embedding(items)
            hidden = self.LayerNorm(hidden)
            hidden = self.dropout(hidden)
            hidden = self.gnns[i](A, hidden)
            alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.hidden_size)
            seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
            seq_hidden = torch.div(torch.sum(seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1), sub_seq_nonz.unsqueeze(1))
            all_sub_sequence.append(seq_hidden)
        all_sub_nonz = torch.stack(all_sub_nonz, dim=1)
        effect_sum = torch.mm(all_sub_nonz.float(), self.effect_weights.unsqueeze(-1))
        alpha = all_sub_nonz.float() * self.effect_weights.unsqueeze(0) / (effect_sum + 1e-8)
        for i in range(self.effect_weights.shape[0]):
            if i==0:
                gnn_output = alpha[:,i].unsqueeze(-1) * all_sub_sequence[i]
            else:
                gnn_output += alpha[:,i].unsqueeze(-1) * all_sub_sequence[i]

        # ---Interest Matching Module
        mixed_x = torch.stack((trm_output, gnn_output.unsqueeze(1).repeat_interleave(item_seq.shape[1], dim=1)), dim=0)
        weights = (torch.matmul(mixed_x, self.attn_weights.unsqueeze(0).unsqueeze(0)) * self.attn).sum(-1)
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        mixed_x = (mixed_x * score).sum(0)
        return mixed_x

        # return trm_output  # [B L H]

    def _get_slice(self, item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in range(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v_1 = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v_1] = 1
                if i > 0:
                    v_2 = np.where(node == u_input[i - 1])[0][0]
                    u_A[u][v_2] = 1
            u_sum = np.sum(u_A, 1)
            u_sum[np.where(u_sum == 0)] = 1
            u_A = u_A / u_sum[:, np.newaxis]
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])


        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(np.array(A)).to(self.device)
        items = torch.LongTensor(np.array(items)).to(self.device)

        return alias_inputs, A, items, mask

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def decrease_nonzero_elements(self,input_tensor):
        mask = input_tensor != 0  # 创建一个与张量相同形状的布尔掩码，标记非零元素
        input_tensor[mask] -= 1  # 仅对掩码为True的元素执行减1操作

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        session_id = interaction['session_id']
        item_type = interaction["item_type_list"]
        last_buy = interaction["item_id"]
        masked_item_seq, pos_items, masked_index, item_type_seq = self.reconstruct_train_data(item_seq, item_type, last_buy)
        self.decrease_nonzero_elements(masked_index)

        seq_output = self.forward(masked_item_seq, item_type_seq)
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]

        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]

        # AW loss
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]
        loss_fct = nn.NLLLoss(reduction='none')
        test_item_emb = self.item_embedding.weight  # [item_num H]

        logits = F.softmax(torch.matmul(seq_output, test_item_emb.transpose(0, 1)).view(-1, test_item_emb.size(0)), dim=1)  # [B mask_len item_num]
        targets = (masked_index > 0).float().view(-1)  # [B*mask_len]
        pos_items = pos_items.view(-1)

        non_zero_mask = targets != 0
        pos_items_non_zero = pos_items[non_zero_mask]
        logits_selected = logits[non_zero_mask, pos_items_non_zero]
        targets[non_zero_mask] = torch.pow((2 - 2 * logits_selected), self.tc)
        loss = torch.sum(loss_fct(torch.log(logits), pos_items) * targets) \
                / torch.sum(targets)

        return loss

    def customized_sort_predict(self, interaction):
        item_seq = interaction['item_id_list']
        type_seq = interaction['item_type_list']
        truth = interaction['item_id']
        if self.dataset == "ijcai_beh":
            raw_candidates = [73, 3050, 22557, 5950, 4391, 6845, 1800, 2261, 13801, 2953, 4164, 32090, 3333, 44733, 7380, 790, 1845, 2886, 2366, 21161, 6512, 1689, 337, 3963, 3108, 715, 169, 2558, 6623, 888, 6708, 3585, 501, 308, 9884, 1405, 5494, 6609, 7433, 25101, 3580, 145, 3462, 5340, 1131, 6681, 7776, 8678, 52852, 19229, 4160, 33753, 4356, 920, 15312, 43106, 16669, 1850, 2855, 43807, 15, 8719, 89, 3220, 36, 2442, 9299, 8189, 701, 300, 526, 4564, 516, 1184, 178, 2834, 16455, 9392, 22037, 344, 15879, 3374, 2984, 3581, 11479, 6927, 779, 5298, 10195, 39739, 663, 9137, 24722, 7004, 7412, 89534, 2670, 100, 6112, 1355]
        elif self.dataset == "retail_beh":
            raw_candidates = [101, 11, 14, 493, 163, 593, 1464, 12, 297, 123, 754, 790, 243, 250, 508, 673, 1161, 523, 41, 561, 2126, 196, 1499, 1093, 1138, 1197, 745, 1431, 682, 1567, 440, 1604, 145, 1109, 2146, 209, 2360, 426, 1756, 46, 1906, 520, 3956, 447, 1593, 1119, 894, 2561, 381, 939, 213, 1343, 733, 554, 2389, 1191, 1330, 1264, 2466, 2072, 1024, 2015, 739, 144, 1004, 314, 1868, 3276, 1184, 866, 1020, 2940, 5966, 3805, 221, 11333, 5081, 685, 87, 2458, 415, 669, 1336, 3419, 2758, 2300, 1681, 2876, 2612, 2405, 585, 702, 3876, 1416, 466, 7628, 572, 3385, 220, 772]
        elif self.dataset == "tmall_beh":
            raw_candidates = [2544, 7010, 4193, 32270, 22086, 7768, 647, 7968, 26512, 4575, 63971, 2121, 7857, 5134, 416, 1858, 34198, 2146, 778, 12583, 13899, 7652, 4552, 14410, 1272, 21417, 2985, 5358, 36621, 10337, 13065, 1235, 3410, 14180, 5083, 5089, 4240, 10863, 3397, 4818, 58422, 8353, 14315, 14465, 30129, 4752, 5853, 1312, 3890, 6409, 7664, 1025, 16740, 14185, 4535, 670, 17071, 12579, 1469, 853, 775, 12039, 3853, 4307, 5729, 271, 13319, 1548, 449, 2771, 4727, 903, 594, 28184, 126, 27306, 20603, 40630, 907, 5118, 3472, 7012, 10055, 1363, 9086, 5806, 8204, 41711, 10174, 12900, 4435, 35877, 8679, 10369, 2865, 14830, 175, 4434, 11444, 701]
        customized_candidates = list()
        for batch_idx in range(item_seq.shape[0]):
            seen = item_seq[batch_idx].cpu().tolist()
            cands = raw_candidates.copy()
            for i in range(len(cands)):
                if cands[i] in seen:
                    new_cand = random.randint(1, self.n_items)
                    while new_cand in seen:
                        new_cand = random.randint(1, self.n_items)
                    cands[i] = new_cand
            cands.insert(0, truth[batch_idx].item())
            customized_candidates.append(cands)
        candidates = torch.LongTensor(customized_candidates).to(item_seq.device)
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self.reconstruct_test_data(item_seq, item_seq_len, type_seq)
        seq_output = self.forward(item_seq, type_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_items_emb = self.item_embedding(candidates)  # delete masked token


        scores = torch.bmm(test_items_emb, seq_output.unsqueeze(-1)).squeeze()  # [B, item_num]


        return scores

