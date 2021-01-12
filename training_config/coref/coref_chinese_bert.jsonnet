// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “Higher-order Coreference Resolution with Coarse-to-fine Inference.” NAACL (2018).
//   + SpanBERT-large

local transformer_model = "bert-base-multilingual-cased"; // "hfl/chinese-bert-wwm-ext"; // "roberta-large";
local max_length = 512;
local feature_size = 20;
local max_span_width = 30;

local transformer_dim = 768;  # uniquely determined by transformer_model
local span_embedding_dim = 3 * transformer_dim + feature_size;
local span_pair_embedding_dim = 3 * span_embedding_dim + feature_size;

local srl = false;
local ner = false;
local ner_sequence = false;

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
    "srl": srl,
    "ner": ner,
    // "srl_per_language": {"english": true, "chinese": false},
    // "ner_per_language": {"english": false, "chinese": false},
    "coref_per_language": {"english": true, "chinese": false},
    // "test_run": true,
    // "limit": 10,
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
    "srl": srl,
    "ner": ner,
    // "srl_per_language": {"english": true, "chinese": false},
    // "ner_per_language": {"english": false, "chinese": false},
    "coref_per_language": {"english": true, "chinese": false},
    "max_span_width": max_span_width,
    // "test_run": true,
    // "limit": 10,
  },
  // "train_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/chinese/train.chinese.v4_gold_conll,/net/nfs2.corp/allennlp/sanjays/coref/train.english.v4_gold_conll",
  "train_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/chinese/train.chinese.v4_gold_conll",
  // "train_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/train.english.v4_gold_conll",
  // "train_data_path": "/net/nfs2.corp/allennlp/sanjays/allennlp-models/serialized_english2/train_chinese_predictions.v4_gold_conll",
  "validation_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/chinese/dev.chinese.v4_gold_conll",
  // "validation_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/dev.english.v4_gold_conll,/net/nfs2.corp/allennlp/sanjays/coref/chinese/dev.chinese.v4_gold_conll",
  // "validation_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/dev.english.v4_gold_conll",
  // "validation_data_path": "/net/nfs2.corp/allennlp/sanjays/allennlp-models/serialized_english2/dev_chinese_predictions.v4_gold_conll",
  // "test_data_path": "/net/nfs2.corp/allennlp/sanjays/coref/chinese/test.chinese.v4_gold_conll",
  "model": {
    "type": "coref_srl",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            // "type": "pretrained_transformer_mismatched_masking",
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
        "dropout": 0.3,
    },
    "antecedent_feedforward": {
        "input_dim": span_pair_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": "relu",
        "dropout": 0.3,
    },
    "srl_scorer": {
        // "input_dim": span_embedding_dim+transformer_dim,
        "input_dim": 2*transformer_dim,
        "num_layers": 2,
        "hidden_dims": transformer_dim,
        "activations": "relu",
        "dropout": 0.3,
    },
    "ner_scorer": {
        // "input_dim": transformer_dim,
        "input_dim": 1500,
        "num_layers": 2,
        // "hidden_dims": transformer_dim/2,
        "hidden_dims": [750, 375],
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
    "spans_per_word": 0.4,
    "max_antecedents": 50,
    "predict_srl": srl,
    "srl_e2e": false,
    "predict_ner": ner,
    "ner_sequence": ner_sequence,
    "consistency_map": {"chinese": true},
    // "language_masking_map": {"english": true, "chinese": true},
    // "only_language_masking_map": {"english": false, "chinese": true},
    "coarse_to_fine": true,
    "inference_order": 2,
    "load_weights_path": "serialized_english2/best.th",
    // "lpsmap": false,
    // "lpsmap_max_iter": 1,
  },
  "data_loader": {
    "type": "coref_dataloader",
    "batch_sampler": {
      // "type": "multilingual",
      // "language_key": "language",
      // "upsample": false,
      // "alternating": true,
      "type": "bucket",
      # Explicitly specifying sorting keys since the guessing heuristic could get it wrong
      # as we a span field.
      "sorting_keys": ["text"],
      "batch_size": 1
    },
    // "probability_of_modified_text": 0.5,
  },
  "trainer": {
    "num_epochs": 4,
    // "patience" : 2,
    "validation_metric": "+coref_f1",
    "num_gradient_accumulation_steps": 2,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-4,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 5e-6}]
      ]
    }
  }
}
