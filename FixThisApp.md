Here are the complete codebase files (attached) of the application I want to run and do my microRNA prediction with.
Can you build me a complete, correct code (UI preferred) that is similar to other executable programs and I can run the whole thing in a single click on the program without going back and forth across the terminal?

I have installed all the dependencies (even updated versions) in Ubuntu (WSL), and I tried to run but it gave me some errors when running because there are some bugs in the code. I want you to find them and fix them. Give me new code (files) so i can then run that in my Ubuntu WSL. 


Only if you can't do this task, then give me an MVP so I can give it to Antigravity and create the thing I need. Please analyze the attached files to debug and resolve these environment-specific issues.


some issues/errors are 
[(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$ python3 MirCleaveNet_predict.py my_data_prepared.txt -m trained_models/MCN.model -o my_results --input_setting 2
{'PAD': 0, 'A(': 1, 'A)': 2, 'AH': 3, 'AM': 4, 'AX': 5, 'AI': 6, 'AB': 7, 'AE': 8, 'C(': 9, 'C)': 10, 'CH': 11, 'CM': 12, 'CX': 13, 'CI': 14, 'CB': 15, 'CE': 16, 'G(': 17, 'G)': 18, 'GH': 19, 'GM': 20, 'GX': 21, 'GI': 22, 'GB': 23, 'GE': 24, 'U(': 25, 'U)': 26, 'UH': 27, 'UM': 28, 'UX': 29, 'UI': 30, 'UB': 31, 'UE': 32, 'N(': 33, 'N)': 34, 'NH': 35, 'NM': 36, 'NX': 37, 'NI': 38, 'NB': 39, 'NE': 40}
2
optimizer: adam
{'name': 'Adam', 'weight_decay': None, 'clipnorm': None, 'global_clipnorm': None, 'clipvalue': None, 'use_ema': False, 'ema_momentum': 0.99, 'ema_overwrite_frequency': None, 'jit_compile': False, 'is_legacy_optimizer': False, 'learning_rate': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}
{'name': 'AdamW', 'learning_rate': 0.001, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False, 'weight_decay': 0.0001, 'exclude_from_weight_decay': None}
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 250)]        0           []

 embedding (Embedding)          (None, 250, 32)      1312        ['input_1[0][0]']

 dropout (Dropout)              (None, 250, 32)      0           ['embedding[0][0]']

 gb_cmodel (GBCmodel)           (None, 250, 32)      2384        ['dropout[0][0]']

 gb_cmodel_1 (GBCmodel)         (None, 250, 32)      2384        ['gb_cmodel[0][0]']

 dropout_1 (Dropout)            (None, 250, 32)      0           ['gb_cmodel_1[0][0]']

 batch_normalization (BatchNorm  (None, 250, 32)     128         ['dropout_1[0][0]']
 alization)

 gb_cmodel_2 (GBCmodel)         (None, 250, 64)      6176        ['batch_normalization[0][0]']

 gb_cmodel_3 (GBCmodel)         (None, 250, 64)      8864        ['gb_cmodel_2[0][0]']

 bidirectional (Bidirectional)  (None, 250, 64)      16640       ['dropout[0][0]']

 dropout_2 (Dropout)            (None, 250, 64)      0           ['gb_cmodel_3[0][0]']

 bidirectional_1 (Bidirectional  (None, 250, 320)    288000      ['bidirectional[0][0]']
 )

 batch_normalization_1 (BatchNo  (None, 250, 64)     256         ['dropout_2[0][0]']
 rmalization)

 hspa (HSPA)                    (None, 250, 320)     205440      ['bidirectional_1[0][0]']

 max_pooling1d (MaxPooling1D)   (None, 250, 64)      0           ['batch_normalization_1[0][0]']

 concatenate (Concatenate)      (None, 250, 384)     0           ['hspa[0][0]',
                                                                  'max_pooling1d[0][0]']

 dropout_3 (Dropout)            (None, 250, 384)     0           ['concatenate[0][0]']

 batch_normalization_2 (BatchNo  (None, 250, 384)    1536        ['dropout_3[0][0]']
 rmalization)

 time_distributed (TimeDistribu  (None, 250, 5)      1925        ['batch_normalization_2[0][0]']
 ted)

==================================================================================================
Total params: 535,045
Trainable params: 534,085
Non-trainable params: 960
__________________________________________________________________________________________________
Traceback (most recent call last):
  File "MirCleaveNet_predict.py", line 257, in <module>
    model.load_weights(parameters["model"]).expect_partial().expect_partial()
  File "/home/dee/miniconda3/envs/mcn/lib/python3.7/site-packages/keras/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/dee/miniconda3/envs/mcn/lib/python3.7/site-packages/keras/optimizers/optimizer_experimental/optimizer.py", line 132, in _create_or_restore_slot_variable
    "You are trying to restore a checkpoint from a legacy Keras "
ValueError: You are trying to restore a checkpoint from a legacy Keras optimizer into a v2.11+ Optimizer, which can cause errors. Please update the optimizer referenced in your code to be an instance of `tf.keras.optimizers.legacy.Optimizer`, e.g.: `tf.keras.optimizers.legacy.Adam`.
(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$]
[[Total params: 535,045
Trainable params: 534,085
Non-trainable params: 960
__________________________________________________________________________________________________
Traceback (most recent call last):
  File "MirCleaveNet_predict.py", line 257, in <module>
    model.load_weights(parameters["model"]).expect_partial()
  File "/home/dee/miniconda3/envs/mcn/lib/python3.7/site-packages/keras/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/dee/miniconda3/envs/mcn/lib/python3.7/site-packages/keras/optimizers/optimizer_experimental/optimizer.py", line 132, in _create_or_restore_slot_variable
    "You are trying to restore a checkpoint from a legacy Keras "
ValueError: You are trying to restore a checkpoint from a legacy Keras optimizer into a v2.11+ Optimizer, which can cause errors. Please update the optimizer referenced in your code to be an instance of `tf.keras.optimizers.legacy.Optimizer`, e.g.: `tf.keras.optimizers.legacy.Adam`.
(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$]]
[(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$ grep -n "elif parameters\[\"optimizer\"\]\|optimizer = optimizers\|optimizer = tf.keras" MirCleaveNet.py
488:        optimizer = optimizers.rmsprop(lr=parameters["learning_rate"], epsilon=parameters["epsilon"])
489:    elif parameters["optimizer"] == 'Adam':
490:        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=parameters["learning_rate"],epsilon=parameters["epsilon"])
492:    elif parameters["optimizer"] == 'nadam':
493:        optimizer = optimizers.nadam(lr=parameters["learning_rate"], epsilon=parameters["epsilon"])
494:    elif parameters["optimizer"] == 'sgd':
495:        optimizer = optimizers.sgd(lr=parameters["learning_rate"])
496:    elif parameters["optimizer"] == 'AdamW':
(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$ cp MirCleaveNet.py MirCleaveNet.py.backup_optimizer_fix2
3 << 'EOF'
with open("MirCleaveNet.py", "r") as f:
    content = f.read()
# Catch the actual lowerca(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$
se 'adam(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$ python3 << 'EOF'
> with open("MirCleaveNet.py", "r") as f:
>     content = f.read()
>
> # Catch the actual lowercase 'adam' branch too, wherever it is
> content = content.replace(
>     'optimizer = parameters["optimizer"]',
  'optim>     'optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=parameters["learning_rate"],epsilon=parameters["epsilon"])'
> )
>
> # Neutralize the debug-only line that creates a non-legacy optimizer as a side effect
ntent = > content = content.replace(
>     "opt1 = tf.keras.optimizers.get('adam')",
>     "opt1 = tf.keras.optimizers.legacy.Adam()"
)
conte> )
> content = content.replace(
   'opt2>     'opt2 = AdamW(learning_rate=1e-3, weight_decay=1e-4)\n    print(opt2.get_config())',
>     '# opt2 debug line removed - AdamW not needed for prediction'
> )
>
> with open("MirCleaveNet.py", "w") as f:
  f.writ>     f.write(content)
>
> print("Patched fallback optimizer branch and debug optimizer lines")
> EOF
Patched fallback optimizer branch and debug optimizer lines
(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$]
[(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$ python3 MirCleaveNet_predict.py my_data_prepared.txt -m trained_models/MCN.model -o my_results --input_setting 2
{'PAD': 0, 'A(': 1, 'A)': 2, 'AH': 3, 'AM': 4, 'AX': 5, 'AI': 6, 'AB': 7, 'AE': 8, 'C(': 9, 'C)': 10, 'CH': 11, 'CM': 12, 'CX': 13, 'CI': 14, 'CB': 15, 'CE': 16, 'G(': 17, 'G)': 18, 'GH': 19, 'GM': 20, 'GX': 21, 'GI': 22, 'GB': 23, 'GE': 24, 'U(': 25, 'U)': 26, 'UH': 27, 'UM': 28, 'UX': 29, 'UI': 30, 'UB': 31, 'UE': 32, 'N(': 33, 'N)': 34, 'NH': 35, 'NM': 36, 'NX': 37, 'NI': 38, 'NB': 39, 'NE': 40}
2
optimizer: <keras.optimizers.optimizer_v2.adam.Adam object at 0x753b37f45350>
{'name': 'Adam', 'learning_rate': 0.001, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 250)]        0           []
 embedding (Embedding)          (None, 250, 32)      1312        ['input_1[0][0]']
 dropout (Dropout)              (None, 250, 32)      0           ['embedding[0][0]']
 gb_cmodel (GBCmodel)           (None, 250, 32)      2384        ['dropout[0][0]']
 gb_cmodel_1 (GBCmodel)         (None, 250, 32)      2384        ['gb_cmodel[0][0]']
 dropout_1 (Dropout)            (None, 250, 32)      0           ['gb_cmodel_1[0][0]']
 batch_normalization (BatchNorm  (None, 250, 32)     128         ['dropout_1[0][0]']
 alization)
 gb_cmodel_2 (GBCmodel)         (None, 250, 64)      6176        ['batch_normalization[0][0]']
 gb_cmodel_3 (GBCmodel)         (None, 250, 64)      8864        ['gb_cmodel_2[0][0]']
 bidirectional (Bidirectional)  (None, 250, 64)      16640       ['dropout[0][0]']
 dropout_2 (Dropout)            (None, 250, 64)      0           ['gb_cmodel_3[0][0]']
 bidirectional_1 (Bidirectional  (None, 250, 320)    288000      ['bidirectional[0][0]']
 )
 batch_normalization_1 (BatchNo  (None, 250, 64)     256         ['dropout_2[0][0]']
 rmalization)
 hspa (HSPA)                    (None, 250, 320)     205440      ['bidirectional_1[0][0]']
 max_pooling1d (MaxPooling1D)   (None, 250, 64)      0           ['batch_normalization_1[0][0]']
 concatenate (Concatenate)      (None, 250, 384)     0           ['hspa[0][0]',
                                                                  'max_pooling1d[0][0]']
 dropout_3 (Dropout)            (None, 250, 384)     0           ['concatenate[0][0]']
 batch_normalization_2 (BatchNo  (None, 250, 384)    1536        ['dropout_3[0][0]']
 rmalization)
 time_distributed (TimeDistribu  (None, 250, 5)      1925        ['batch_normalization_2[0][0]']
 ted)
==================================================================================================
Total params: 535,045
Trainable params: 534,085
Non-trainable params: 960
__________________________________________________________________________________________________
Traceback (most recent call last):
  File "MirCleaveNet_predict.py", line 257, in <module>
    model.load_weights(parameters["model"]).expect_partial()
  File "/home/dee/miniconda3/envs/mcn/lib/python3.7/site-packages/keras/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/dee/miniconda3/envs/mcn/lib/python3.7/site-packages/tensorflow/python/training/saving/saveable_object_util.py", line 142, in restore
    f"and name {self.name}.") from e
ValueError: Received incompatible tensor with shape (32, 256) when attempting to restore variable with shape (32, 128) and name variables/109/.ATTRIBUTES/VARIABLE_VALUE.
(mcn) dee@DAWOOD-DELL:~/mircleavenet_work/MirCleaveNet-main$]
[and so on without giving output]
