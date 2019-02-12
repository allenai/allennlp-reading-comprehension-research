To run the demo for the augmented QANet model,
 you should clone the [`allennlp-demo` ](https://github.com/allenai/allennlp-demo) repo
 and replace the `server/models.py` and `demo/src/components/demos/MachineComprehension.js` 
 with the corresponding files in this folder.
 
Note that we just reuse the `machine_comprehension` predictor in the allennlp library, 
 but that predictor uses the `"dataset_reader"` as key to initiate from parameters 
 [(this line)](https://github.com/allenai/allennlp/blob/39413f220f2e682001299f8975a38f2221fac840/allennlp/predictors/predictor.py#L126),
 while we actually use `"validation_dataset_reader"` as the key for evaluation data reader in the archived model file. This could cause 
 errors. To solve this issue, you can either change that line in the allennlp library, or implement your own predictor. The easiest way 
 is to just unzip the model `tar.gz` file, change the `"validation_dataset_reader"` field to `"dataset_reader"` field in the 
 `config.json` file, and then re-compress those files.