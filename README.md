# cs685_proj
Exploration of attribute transfer techniques for political news depolarization


A) Reverse attention code :

reverse_attention_implementation.ipynb provides the partial implementation for the reverse attention code. xlnet_train.py provides the code for pretraining a Xlnet classifier required for the first step of the reverse attention.


B) Paraphrase Model Code:

The main repository can be found at https://github.com/martiansideofthemoon/style-transfer-paraphrase. We changed the following files for our code:

  a. Conversion to binary files:
  bpe2binary.sh converts the dataset txt and label files to bpe format needed for finetuning.

  b. Finetuning parapharsing model:
  run_finetuning_lm.py is used to run finetuning on paraphrase model using our dataset.

  c. Baseline preprocess:
  baseline_preprocess.py is used to tokenize our dataset for the baseline model.
  
  
C) Error Analysis.csv:
 This file contains the error analysis on 100 randomly chosen sentences that failed to convert to the target class. The final column indicates whether the sentences before and after attribute transfer remain the same.
 
D) Evaluation Files for the Model

  1. roberta_classify.py : This is the evaluation code for the roberta classifier run on the model. 
  2. acceptability.py : Consists of evaluation code for analysing the fluency metric on generated samples. 
  3. get_paraphrase_similarity.py : This file consists of the evaluation code for computing the similarity between the input sentences and the generated paraphrased   sentences. 


