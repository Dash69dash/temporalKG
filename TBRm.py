class TransR(PairwiseModel):
    def __init__(self, **kwargs):
        super(TransR, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "rel_hidden_size", "ent_hidden_size", "l1_flag"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        self.ent_embeddings = NamedEmbedding("ent_embedding", self.tot_entity, self.ent_hidden_size)
        self.rel_embeddings = NamedEmbedding("rel_embedding", self.tot_relation, self.rel_hidden_size)
        self.rel_matrix = NamedEmbedding("rel_matrix", self.tot_relation, self.ent_hidden_size * self.rel_hidden_size)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_matrix.weight)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
            self.rel_matrix,
        ]

        self.loss = Criterion.pairwise_hinge
    
    def transform(self, e, matrix):
        matrix = matrix.view(-1, self.ent_hidden_size, self.rel_hidden_size)
        if e.shape[0] != matrix.shape[0]:
            e = e.view(-1, matrix.shape[0], self.ent_hidden_size).permute(1, 0, 2)
            e = torch.matmul(e, matrix).permute(1, 0, 2)
        else:
            e = e.view(-1, 1, self.ent_hidden_size)
            e = torch.matmul(e, matrix)
        return e.view(-1, self.rel_hidden_size)

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        t_e = self.ent_embeddings(t)

        h_e = F.normalize(h_e, p=2, dim=-1)
        r_e = F.normalize(r_e, p=2, dim=-1)
        t_e = F.normalize(t_e, p=2, dim=-1)

        h_e = torch.unsqueeze(h_e, 1)
        t_e = torch.unsqueeze(t_e, 1)
        # [b, 1, k]

        matrix = self.rel_matrix(r)
        # [b, k, d]

        transform_h_e = self.transform(h_e, matrix)
        transform_t_e = self.transform(t_e, matrix)
        # [b, 1, d] = [b, 1, k] * [b, k, d]

        h_e = torch.squeeze(transform_h_e, axis=1)
        t_e = torch.squeeze(transform_t_e, axis=1)
        # [b, d]
        return h_e, r_e, t_e

    def forward(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids.
               t (Tensor): Tail entity ids.

            Returns:
                Tensors: the scores of evaluationReturns head, relation and tail embedding Tensors.
        """
        h_e, r_e, t_e = self.embed(h, r, t)

        norm_h_e = F.normalize(h_e, p=2, dim=-1)
        norm_r_e = F.normalize(r_e, p=2, dim=-1)
        norm_t_e = F.normalize(t_e, p=2, dim=-1)

        if self.l1_flag:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=1, dim=-1)

        return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=2, dim=-1)