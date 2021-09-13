""" The training parameters are set in accordance with the setup we used for BERT
    Feel free to change parameters as you see fit
    Use the parameters described in our paper to reproduce results for other models (XLM RoBERTA and DistilBERT)"""

model_type = "bert"
# model_type = "xlmroberta"
# model_type = "distilbert"
model_name = "bert-base-multilingual-cased"
# model_name = "xlm-roberta-base"
# model_name = "distilbert-base-multilingual-cased"

train_args = {
        "max_seq_length": 512,
        "num_train_epochs": 15,
        "fp16" : True,
        # "scheduler" : "polynomial_decay_schedule_with_warmup",
        # "polynomial_decay_schedule_lr_end": 1e-7,
        # "polynomial_decay_schedule_power": 2.0,
        "scheduler" : "cosine_schedule_with_warmup",
        "cosine_schedule_num_cycles": 0.5,
        #"scheduler" : "constant_schedule",
        "learning_rate": 1e-5,
        "train_batch_size": 8,
        "weight_decay": 1,
        "reprocess_input_data": False,
        "overwrite_output_dir": True,
        "use_cached_eval_features": False,
        "no_save": False,
        "use_early_stopping": False,
        "evaluate_during_training": False,
        #the following two parameters are for when this script is run from google colab and wandb is connected with colab
        # "wandb_project": "BERT_final", #change here for each separate run
        # "wandb_kwargs": {"name": "final_"+str(FoldIndex),"entity" :'fakenewscovid'}, #change here for each separate run
        "save_model_every_epoch": False,
        "save_eval_checkpoints": False,
        "evaluate_during_training_verbose" : True
    }