from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from tqdm import tqdm
import ot
from math import log
from collections import defaultdict, Counter
from transformers import AutoModelForMaskedLM, AutoTokenizer
import datasets

_CITATION = """\
@inproceedings{colombo-etal-2021-automatic,
    title = "Automatic Text Evaluation through the Lens of {W}asserstein Barycenters",
    author = "Colombo, Pierre  and Staerman, Guillaume  and Clavel, Chlo{\'e}  and Piantanida, Pablo",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year = "2021",
    pages = "10450--10466"
}
"""

_DESCRIPTION = """\

"""

_KWARGS_DESCRIPTION = """
Computes BaryScore score of a candidate text against one reference text.
Args:
    predictions: list of translations to score.
        Each  should be .
    references: list of lists of references for each translation.
        Each reference should be tokenized into a list of tokens.
Returns:
    'baryscore_W': baryscore score,
    'baryscore_SD': baryscore score,

Examples:
    >>> predictions = [
    ...       'I like my cakes very much',                       # the first sample
    ...       'I hate these cakes!',                             # the second sample
    ... ]
    
    >>> references = [
    ...       'I like my cakes very much',                       # the first references
    ...       'I like my cakes very much',                       # the second references
    ... ]
    >>> bary_score = datasets.load_metric("bary_score")
    >>> results = bary_score.compute(predictions=predictions, references=references)
    >>> print(results)
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BaryScoreMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/PierreColombo/nlg_eval_via_simi_measures"],
            reference_urls=[
                "https://arxiv.org/abs/2108.12463"
            ],
        )

    def prepare_idfs(self, hyps, refs):
        """
        :param hyps: hypothesis list of string sentences has to be computed at corpus level
        :param refs:reference list of string sentences has to be computed at corpus level
        """
        t_hyps = self.tokenizer(hyps)['input_ids']
        t_refs = self.tokenizer(refs)['input_ids']
        idf_dict_ref = self.ref_list_to_idf(t_refs)
        idf_dict_hyp = self.ref_list_to_idf(t_hyps)
        idfs_tokenizer = (idf_dict_ref, idf_dict_hyp)
        self.model_ids = idfs_tokenizer
        return idf_dict_hyp, idf_dict_ref

    def ref_list_to_idf(self, input_refs):
        """
        :param input_refs: list of input reference
        :return: idf dictionnary
        """
        idf_count = Counter()
        num_docs = len(input_refs)

        idf_count.update(sum([list(set(i)) for i in input_refs], []))

        idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
        idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
        return idf_dict

    def load_tokenizer_and_model(self):
        """
        Loading and initializing the chosen model and tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained('{}'.format(self.model_name))
        model = AutoModelForMaskedLM.from_pretrained('{}'.format(self.model_name))
        model.config.output_hidden_states = True
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def _compute(self, predictions, references, idf_hyps=None, idf_ref=None, model_name="bert-base-uncased",
                 last_layers=5, use_idfs=True, sinkhorn_ref=0.01):
        """
        :param predictions: hypothesis list of string sentences
        :param references: reference list of string sentences
        :param idf_hyps: idfs of hypothesis computed at corpus level
        :param idf_ref: idfs of references computed at corpus level
        :param model_name: model name or path from HuggingFace Librairy
        :param last_layers: last layer to use in the pretrained model
        :param use_idfs: if true use idf costs else use uniform weights
        :param sinkhorn_ref:  weight of the KL in the SD
        :return: dictionnary of scores
        """

        self.model_name = model_name
        self.load_tokenizer_and_model()
        n = self.model.config.num_hidden_layers + 1
        assert n - last_layers > 0
        self.layers_to_consider = range(n - last_layers, n)
        self.use_idfs = use_idfs
        self.sinkhorn_ref = sinkhorn_ref
        self.idfs = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ###############################################
        ## Extract Embeddings From Pretrained Models ##
        ###############################################
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
        nb_sentences = len(references)
        baryscores = []
        assert len(predictions) == len(references)

        if (idf_hyps is None) and (idf_ref is None):
            idf_hyps, idf_ref = self.model_ids

        model = self.model.to(self.device)

        with torch.no_grad():
            ###############################################
            ## Extract Embeddings From Pretrained Models ##
            ###############################################
            batch_refs = self.tokenizer(references, return_tensors='pt', padding=True).to(self.device)
            batch_refs_embeddings_ = model(**batch_refs)[-1]

            batch_hyps = self.tokenizer(predictions, return_tensors='pt', padding=True).to(self.device)
            batch_hyps_embeddings_ = model(**batch_hyps)[-1]

            batch_refs_embeddings = [batch_refs_embeddings_[i] for i in list(self.layers_to_consider)]
            batch_hyps_embeddings = [batch_hyps_embeddings_[i] for i in list(self.layers_to_consider)]

            batch_refs_embeddings = torch.cat([i.unsqueeze(0) for i in batch_refs_embeddings])
            batch_refs_embeddings.div_(torch.norm(batch_refs_embeddings, dim=-1).unsqueeze(-1))
            batch_hyps_embeddings = torch.cat([i.unsqueeze(0) for i in batch_hyps_embeddings])
            batch_hyps_embeddings.div_(torch.norm(batch_hyps_embeddings, dim=-1).unsqueeze(-1))

            ref_tokens_id = batch_refs['input_ids'].cpu().tolist()
            hyp_tokens_id = batch_hyps['input_ids'].cpu().tolist()

            ####################################
            ## Unbatched BaryScore Prediction ##
            ####################################
            for index_sentence in tqdm(range(nb_sentences), 'BaryScore Progress'):
                dict_score = {}
                ref_ids_idf = batch_refs['input_ids'][index_sentence]
                hyp_idf_ids = batch_hyps['input_ids'][index_sentence]

                ref_tokens = [i for i in self.tokenizer.convert_ids_to_tokens(ref_tokens_id[index_sentence],
                                                                              skip_special_tokens=False) if
                              i != self.tokenizer.pad_token]
                hyp_tokens = [i for i in self.tokenizer.convert_ids_to_tokens(hyp_tokens_id[index_sentence],
                                                                              skip_special_tokens=False) if
                              i != self.tokenizer.pad_token]

                ref_ids = [k for k, w in enumerate(ref_tokens) if True]
                hyp_ids = [k for k, w in enumerate(hyp_tokens) if True]

                # With stop words
                ref_idf_i = [idf_ref[i] for i in ref_ids_idf[ref_ids]]
                hyp_idf_i = [idf_hyps[i] for i in hyp_idf_ids[hyp_ids]]

                ref_embedding_i = batch_refs_embeddings[:, index_sentence, ref_ids, :]
                hyp_embedding_i = batch_hyps_embeddings[:, index_sentence, hyp_ids, :]
                measures_locations_ref = ref_embedding_i.permute(1, 0, 2).cpu().numpy().tolist()
                measures_locations_ref = [np.array(i) for i in measures_locations_ref]
                measures_locations_hyps = hyp_embedding_i.permute(1, 0, 2).cpu().numpy().tolist()
                measures_locations_hyps = [np.array(i) for i in measures_locations_hyps]

                if self.use_idfs:
                    #########################
                    ## Use TF-IDF weights  ##
                    #########################
                    baryscore = self.baryscore(measures_locations_ref, measures_locations_hyps, ref_idf_i,
                                               hyp_idf_i)
                else:
                    #####################
                    ## Uniform Weights ##
                    #####################
                    uniform_refs = [1 / len(measures_locations_ref)] * len(measures_locations_ref)
                    uniform_hyps = [1 / len(measures_locations_hyps)] * len(measures_locations_hyps)
                    baryscore = self.baryscore(measures_locations_ref, measures_locations_hyps, uniform_refs,
                                               uniform_hyps)

                for key, value in baryscore.items():
                    dict_score['baryscore_{}'.format(key)] = value
                baryscores.append(dict_score)
            baryscores_dic = {}
            for k in dict_score.keys():
                baryscores_dic[k] = []
                for score in baryscores:
                    baryscores_dic[k].append(score[k])

        return baryscores_dic

    def baryscore(self, measures_locations_ref, measures_locations_hyps, weights_refs, weights_hyps):
        """
        :param measures_locations_ref: input measure reference locations
        :param measures_locations_hyps: input measure hypothesis locations
        :param weights_refs: references weights in the Wasserstein Barycenters
        :param weights_hyps: hypothesis weights in the Wasserstein Barycenters
        :return:
        """

        weights_hyps = np.array([i / sum(weights_hyps) for i in weights_hyps]).astype(np.float64)
        weights_refs = np.array([i / sum(weights_refs) for i in weights_refs]).astype(np.float64)

        self.n_layers = measures_locations_ref[0].shape[0]
        self.d_bert = measures_locations_ref[0].shape[1]
        ####################################
        ## Compute Wasserstein Barycenter ##
        ####################################
        bary_ref = self.w_barycenter(measures_locations_ref, weights_refs)
        bary_hyp = self.w_barycenter(measures_locations_hyps, weights_hyps)

        #################################################
        ## Compute Wasserstein and Sinkhorn Divergence ##
        #################################################

        C = ot.dist(bary_ref, bary_hyp)
        weights_first_barycenter = np.zeros((C.shape[0])) + 1 / C.shape[0]
        weights_second_barycenter = np.zeros((C.shape[1])) + 1 / C.shape[1]
        wasserstein_distance = ot.emd2(weights_first_barycenter, weights_second_barycenter, C,
                                       log=True)[0]
        wasserstein_sinkhorn = ot.bregman.sinkhorn2(weights_first_barycenter, weights_second_barycenter, C,
                                                    reg=self.sinkhorn_ref, numItermax=10000).tolist()
        return {
            "W": wasserstein_distance,
            "SD": wasserstein_sinkhorn
        }

    def w_barycenter(self, measures_locations, weights):
        """
        :param measures_locations: location of the discrete input measures
        :param weights: weights of the input measures
        :return: barycentrique distribution
        """
        X_init = np.zeros((self.n_layers, self.d_bert))
        m = np.zeros((self.n_layers)) + 1 / self.n_layers
        measures_weights = [m] * self.n_layers
        mesure_bary = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init,
                                                    b=m.copy(),
                                                    weights=weights,
                                                    numItermax=1000, verbose=False)
        return mesure_bary


if __name__ == '__main__':
    predictions = [
        'I like my cakes very much',  # the first sample
        'I hate these cakes!',  # the second sample
    ]
    references = [
        'I like my cakes very much',  # the first references
        'I like my cakes very much',  # the second references
    ]
    bary_score = datasets.load_metric("bary_score")
    results = bary_score.compute(predictions=predictions, references=references)
    print(results)
