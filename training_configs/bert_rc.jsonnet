{
    "dataset_reader": {
        "type": "drop",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased",
                "do_lowercase": true,
                "use_starting_offsets": false
            },
        },
        "passage_length_limit": 300,
        "question_length_limit": 30,
        "skip_when_all_empty": ["passage_span"],
        "instance_format": "bert"
    },
    "validation_dataset_reader": {
        "type": "drop",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased",
                "do_lowercase": true,
                "use_starting_offsets": false
            },
        },
        "passage_length_limit": 400,
        "question_length_limit": 30,
        "skip_when_all_empty": [],
        "instance_format": "bert"
    },
    "train_data_path": std.extVar("DROP_TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("DROP_DEV_DATA_PATH"),
    "model": {
        "type": "bert_rc_marginal",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets", "segment_labels"],
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "bert-base-uncased",
                    "requires_grad": true,
                    "top_layer_only": true
                },
            }
        },
        "dropout": 0.0,
        "regularizer": [
            [
                ".*",
                {
                    "type": "l2",
                    "alpha": 1e-07
                }
            ]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "passage",
                "num_tokens"
            ],
            [
                "question",
                "num_tokens"
            ]
        ],
        "batch_size": 10,
        "max_instances_in_memory": 600
    },
    "trainer": {
        "num_epochs": 10,
        "grad_norm": 5,
        "patience": 5,
        "validation_metric": "+f1",
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 3e-5,
            "betas": [
                0.9,
                0.999
            ],
        },
    }
}
