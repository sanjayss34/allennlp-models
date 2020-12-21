// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “Higher-order Coreference Resolution with Coarse-to-fine Inference.” NAACL (2018).
//   + SpanBERT-large

local transformer_model = "bert-base-cased"; // "SpanBERT/spanbert-large-cased"; // "roberta-large";
local max_length = 512;
local feature_size = 20;
local max_span_width = 30;

local transformer_dim = 768;  # uniquely determined by transformer_model
local span_embedding_dim = 3 * transformer_dim + feature_size;
local span_pair_embedding_dim = 3 * span_embedding_dim + feature_size;

local pickle_root = "/net/nfs2.corp/allennlp/sanjays/coref/pickle/";

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
    "max_sentences": 110,
    "srl": true,
    "pickle_path": pickle_root+"conll_train_coref_srl_"+transformer_model+".pkl",
    "test_run": true,
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
    "srl": true,
    "pickle_path": pickle_root+"conll_dev_coref_srl_"+transformer_model+".pkl",
    "test_run": true,
  },
  "train_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/train.english.v4_gold_conll",
  "validation_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/dev.english.v4_gold_conll",
  // "test_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/test.english.v4_gold_conll",
  "model": {
    "type": "coref_srl",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": transformer_model,
            "max_length": max_length
        }
      }
    },
    "context_layer": {
        "type": "pass_through",
        "input_dim": transformer_dim
    },
    "mention_feedforward": {
        "input_dim": span_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": "relu",
        "dropout": 0.3
    },
    "antecedent_feedforward": {
        "input_dim": span_pair_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": "relu",
        "dropout": 0.3
    },
    "srl_predicate_scorer": {
        "input_dim": transformer_dim,
        "num_layers": 2,
        "hidden_dims": [transformer_dim/2, 1],
        "activations": ["relu", "linear"],
        "dropout": 0.3,
    },
    "srl_scorer": {
        "input_dim": span_embedding_dim+transformer_dim,
        "num_layers": 1,
        "hidden_dims": (transformer_dim+span_embedding_dim)/2,
        "activations": "relu",
        "dropout": 0.3,
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
    "spans_per_word": 0.8,
    "srl_predicate_candidates_per_word": 0.4,
    "predict_srl": true,
    "predict_coref": false,
    "max_antecedents": 50,
    "coarse_to_fine": true,
    "inference_order": 2,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      # Explicitly specifying sorting keys since the guessing heuristic could get it wrong
      # as we a span field.
      "sorting_keys": ["text"],
      "batch_size": 1
    }
  },
  "trainer": {
    "num_epochs": 40,
    "patience" : 10,
    "validation_metric": "+coref_f1",
    "cuda_device": -1,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-4,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    }
  }
}
