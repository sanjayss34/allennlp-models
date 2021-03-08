// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “Higher-order Coreference Resolution with Coarse-to-fine Inference.” NAACL (2018).
//   + SpanBERT-large

local transformer_model = "bert-base-multilingual-cased"; // "roberta-large";
local max_length = 512;
local feature_size = 20;
local max_span_width = 30;

local transformer_dim = 768;  # uniquely determined by transformer_model
local lstm_dim = 200;
local span_embedding_dim = 3 * transformer_dim + feature_size;
// local span_embedding_dim = 4 * lstm_dim + transformer_dim + feature_size;
local span_pair_embedding_dim = 3 * span_embedding_dim + feature_size;

local pickle_root = "/net/nfs2.reserved/allennlp/sanjays/coref/pickle/";
local srl = false;
local ner = true;
local ner_sequence = true;

local dropout = 0.3; // std.parseJson(std.extVar("dropout"));
local alignment_attention = { // if std.extVar("parallel_alignment_attention") == "dot_product" then {
   "type": "dot_product"
}; //  else {
//    "type": "bilinear",
//    "matrix_1_dim": transformer_dim,
//    "matrix_2_dim": transformer_dim
// };


{
  "dataset_reader": {
    "type": "coref",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length
      },
    },
    "max_span_width": max_span_width,
    // "max_sentences": 110,
    "individual_sentences": true,
    "srl": srl,
    "ner": ner,
    // "pickle_path": pickle_root+"conll_train_coref_srl_ignore_predicate_spans2.pkl",
    // "test_run": true,
    "parallel_jieba": true,
    "limit": 200000,
  },
  "validation_dataset_reader": {
    "type": "coref",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length
      },
    },
    "max_span_width": max_span_width,
    "individual_sentences": true,
    "srl": srl,
    "ner": ner,
    // "pickle_path": pickle_root+"conll_train_coref_srl_ignore_predicate_spans2.pkl",
    // "test_run": true,
    // "limit": 20,
  },
  "train_data_path": "/net/nfs2.reserved/allennlp/sanjays/coref/train.english.v4_gold_conll,/home/sanjays/UM-Corpus/data/Bilingual/News/Bi_news-parallel.txt", // /net/nfs2.reserved/allennlp/sanjays/coref/chinese/train.chinese.v4_gold_conll",
  "validation_data_path": "/net/nfs2.reserved/allennlp/sanjays/coref/dev.english.v4_gold_conll",
  // "train_data_path": "/data/english/train.english.v4_gold_conll,/parallel/Bi_news-parallel.txt",
  // "validation_data_path": "/data/english/dev.english.v4_gold_conll",
  // "test_data_path": "/net/nfs2.reserved/allennlp/sanjays/coref/test.english.v4_gold_conll",
  "model": {
    "type": "coref_srl",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": transformer_model,
            "max_length": max_length,
            // "train_parameters": false,
        }
      }
    },
    "context_layer": {
        "type": "pass_through",
        "input_dim": transformer_dim
    },
    // "context_layer": {
    //     "type": "lstm",
    //     "bidirectional": true,
    //     "hidden_size": lstm_dim,
    //     "input_size": transformer_dim,
    //     "num_layers": 2,
    //     "dropout": 0.3,
    // },
    "mention_feedforward": {
        "input_dim": span_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": "relu",
        "dropout": dropout
    },
    "antecedent_feedforward": {
        "input_dim": span_pair_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": "relu",
        "dropout": dropout
    },
    "srl_predicate_scorer": {
        "input_dim": transformer_dim,
        "num_layers": 2,
        "hidden_dims": [transformer_dim/2, 1],
        "activations": ["relu", "linear"],
        "dropout": dropout,
    },
    "srl_scorer": {
        "input_dim": 2*transformer_dim,
        // "input_dim": 4*lstm_dim,
        "num_layers": 2,
        "hidden_dims": [3*transformer_dim/2, transformer_dim],
        // "hidden_dims": 2*lstm_dim,
        "activations": "relu",
        "dropout": dropout,
    },
    "ner_scorer": {
        "input_dim": transformer_dim,
        // "input_dim": 1500,
        "num_layers": 1,
        "hidden_dims": transformer_dim,
        // "hidden_dims": 750,
        "activations": "relu",
        "dropout": dropout,
    },
    "initializer": {
      "regexes": [
        [".*_span_updating_gated_sum.*weight", {"type": "xavier_normal"}],
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer.*weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
      ]
    },
    "feature_size": feature_size,
    "max_span_width": max_span_width,
    "spans_per_word": 0.4,
    "srl_predicate_candidates_per_word": 0.4,
    "predict_srl": srl,
    "srl_e2e": false,
    "predict_ner": ner,
    "ner_sequence": ner_sequence,
    "mention_score_loss": true,
    "predict_coref": false,
    "max_antecedents": 50,
    "coarse_to_fine": true,
    "inference_order": 2,
    "parallel_consistency": true,
    "parallel_consistency_coefficient": 1.0, // std.parseJson(std.extVar("parallel_consistency_coefficient")),
    "parallel_alignment_activation": "softmax", // std.extVar("parallel_alignment_activation"),
    "parallel_alignment_attention": alignment_attention,
    "parallel_alignment_agreement_coefficient": 0.0,
    "parallel_lpsmap": false,
    "parallel_asymmetric": true,
  },
  "data_loader": {
    // "type": "coref_dataloader",
    "batch_sampler": {
      // "type": "bucket",
      "type": "multilingual",
      "upsample": false,
      "alternating_fixed": true,
      # Explicitly specifying sorting keys since the guessing heuristic could get it wrong
      # as we a span field.
      "sorting_keys": ["text"],
      "batch_size": 8
    },
    // "probability_of_modified_text": 1,
    // "language_map": {"english": false, "chinese": true},
  },
  "trainer": {
    "num_epochs": 40,
    "patience" : 10,
    "validation_metric": "+ner_f1",
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "num_gradient_accumulation_steps": 2,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-4, // std.parseJson(std.extVar("lr")),
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}], // std.parseJson(std.extVar("bert_lr"))}],
        // [[".*transformer.embeddings.*"], {"lr": 0}]
      ]
    }
  }
}
