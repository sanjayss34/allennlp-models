local num_epochs = 5;
local batch_size = 16;
local model_name = "bert-base-cased";
local label_namespace = "label_other";

{
    "dataset_reader": {
        "type": "srl",
        "bert_model_name": model_name,
        // "limit": 8341,
        // "limit": 100,
        // "random_sample": true,
        // "random_seed": 98203,
        "mismatched_tokens": true,
        "label_namespace": label_namespace
        // "type": "srl_conll2005",
        // "text_root": "/net/nfs.corp/allennlp/sanjays/treebank_3/raw/wsj/"
    },
    "validation_dataset_reader": {
        "type": "srl",
        "bert_model_name": model_name,
        // "limit": 1152,
        // "random_sample": true,
        // "random_seed": 98203,
        "mismatched_tokens": true,
        "label_namespace": label_namespace
        // "limit": 100,
        // "type": "srl_conll2005",
        // "text_root": "/net/nfs.corp/allennlp/sanjays/treebank_3/raw/wsj/"
    },
    "model": {
        "type": "srl_bert",
        "bert_model": model_name,
        "mismatched_embedder": {
            "type": "pretrained_transformer_mismatched",
            "model_name": model_name
        },
        // "encoder": {
        //     "type": "lstm",
        //     "input_size": 778,
        //     "hidden_size": 768,
        //     "bidirectional": true,
        // },
        // "mlp_hidden_size": 300,
        "lpsmap": true,
        "lpsmap_core_roles_only": true,
        "constrain_crf_decoding": false,
        "label_encoding": "BIO",
        "batch_size": batch_size,
        "reinitialize_pos_embedding": false,
        "embedding_dropout": 0.1,
        "label_namespace": label_namespace
    },
    "train_data_path": "/net/nfs.corp/allennlp/data/ontonotes/conll-formatted-ontonotes-5.0/data/train",
    "validation_data_path": "/net/nfs.corp/allennlp/data/ontonotes/conll-formatted-ontonotes-5.0/data/development",
    // "train_data_path": "/net/nfs.corp/allennlp/sanjays/conll2005/conll05st-release/train/props/",
    // "validation_data_path": "/net/nfs.corp/allennlp/sanjays/conll2005/conll05st-release/devel/props/",
    "trainer": {
        "cuda_device": 0,
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },
        // "grad_norm": 1,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            // "num_epochs": num_epochs,
            // "warmup_steps": num_epochs*8689/10,
            // "end_learning_rate": 5e-5
        },
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "huggingface_adamw",
            "correct_bias": false,
            "lr": 5e-05,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.bias",
                        "LayerNorm.weight",
                        "layer_norm.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "weight_decay": 0.01
        },
        "validation_metric": "+f1-measure-overall"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        }
    }
}
