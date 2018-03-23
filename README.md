# One-Shot Visual Imitation Learning via Meta-Learning

*A TensorFlow implementation of the paper [One-Shot Visual Imitation Learning via Meta-Learning (Finn*, Yu* et al., 2017)](https://arxiv.org/pdf/1709.04905.pdf).* Here are the instructions to run our experiments shown in the paper.

First clone the fork of the gym repo found [here](https://github.com/tianheyu927/gym), and following the instructions there to install gym. Switch to branch *mil*.

Then go to the `mil` directory and run `./scripts/get_data.sh` to download the data.

After downloading the data, training and testing scripts for MIL are available in `scripts/`.

*Note: The code only includes the simulated experiments.*


错误文档3.22（./scripts/run_sim_push.sh）
Traceback (most recent call last):
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 510, in _apply_op_helper
    preferred_dtype=default_dtype)
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1036, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/constant_op.py", line 235, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/constant_op.py", line 214, in constant
    value, dtype=dtype, shape=shape, verify_shape=verify_shape))
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/tensor_util.py", line 442, in make_tensor_proto
    _GetDenseDimensions(values)))
ValueError: Argument must be a dense tensor: range(0, 20) - got shape [20], but wanted [].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 524, in _apply_op_helper
    values, as_ref=input_arg.is_ref).dtype.name
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1036, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/constant_op.py", line 235, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/constant_op.py", line 214, in constant
    value, dtype=dtype, shape=shape, verify_shape=verify_shape))
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/tensor_util.py", line 442, in make_tensor_proto
    _GetDenseDimensions(values)))
ValueError: Argument must be a dense tensor: range(0, 20) - got shape [20], but wanted [].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "main.py", line 295, in <module>
    main()
  File "main.py", line 261, in main
    model.init_network(graph, input_tensors=train_input_tensors, restore_iter=FLAGS.restore_iter)
  File "/home/lip/mil/mil.py", line 38, in init_network
    network_config=self.network_params)
  File "/home/lip/mil/mil.py", line 574, in construct_model
    unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0]))
  File "/home/lip/mil/mil.py", line 460, in batch_metalearn
    local_outputa, final_eept_preda = self.forward(inputa, state_inputa, weights, network_config=network_config)
  File "/home/lip/mil/mil.py", line 225, in forward
    context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), range(FLAGS.bt_dim)))
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/ops/array_ops.py", line 2667, in gather
    params, indices, validate_indices=validate_indices, name=name)
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/ops/gen_array_ops.py", line 1777, in gather
    validate_indices=validate_indices, name=name)
  File "/home/lip/.pyenv/versions/3.5.4/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 528, in _apply_op_helper
    (input_name, err))
ValueError: Tried to convert 'indices' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 20) - got shape [20], but wanted [].

