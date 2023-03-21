import os
from typing import List

from absl import app as absl_app
from absl import flags
from absl import logging
import tensorflow as tf

from data import dataset
from decoder import DeepSpeechDecoder
from deep_speech_model import DeepSpeech2
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


_VOCABULARY_FILE = os.path.join(os.path.dirname(__file__), "data/vocabulary.txt")
_WER_KEY = "WER"
_CER_KEY = "CER"


def compute_length_after_conv(max_time_steps, ctc_time_steps, input_length):
    ctc_input_length = tf.cast(tf.multiply(input_length, ctc_time_steps), dtype=tf.float32)
    return tf.cast(tf.math.floordiv(ctc_input_length, tf.cast(max_time_steps, dtype=tf.float32)), dtype=tf.int32)


def evaluate_model(estimator, speech_labels, entries, input_fn_eval):
    predictions = estimator.predict(input_fn=input_fn_eval)
    probs = [pred["probabilities"] for pred in predictions]

    num_of_examples = len(probs)
    targets = [entry[2] for entry in entries]

    total_wer, total_cer = 0, 0
    greedy_decoder = DeepSpeechDecoder(speech_labels)
    for i in range(num_of_examples):
        decoded_str = greedy_decoder.decode(probs[i])
        total_cer += greedy_decoder.cer(decoded_str, targets[i]) / float(len(targets[i]))
        total_wer += greedy_decoder.wer(decoded_str, targets[i]) / float(len(targets[i].split()))

    total_cer /= num_of_examples
    total_wer /= num_of_examples

    global_step = estimator.get_variable_value(tf.compat.v1.GraphKeys.GLOBAL_STEP)
    eval_results = {
        _WER_KEY: total_wer,
        _CER_KEY: total_cer,
        tf.compat.v1.GraphKeys.GLOBAL_STEP: global_step,
    }

    return eval_results


def model_fn(features, labels, mode, params):
    num_classes = params["num_classes"]
    input_length = features["input_length"]
    label_length = features["label_length"]
    features = features["features"]

    model = DeepSpeech2(
        flags_obj.rnn_hidden_layers, flags_obj.rnn_type,
        flags_obj.is_bidirectional, flags_obj.rnn_hidden_size,
        num_classes, flags_obj.use_bias)

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(features, training=False)
        predictions = {
            "classes": tf.argmax(logits, axis=2),
            "probabilities": logits,
            "logits": logits
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    logits = model(features, training=True)
    ctc_input_length = compute_length_after_conv(tf.shape(features)[1], tf.shape(logits)[1], input_length)
    loss = tf.reduce_mean(tf.keras.backend.ctc_batch_cost(labels, logits, ctc_input_length, label_length))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=flags_obj.learning_rate)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    minimize_op = optimizer.minimize(loss, global_step=global_step)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    print('Saving Keras Model')
    keras_model = tf.keras.Model(inputs=features, outputs=logits)
    keras_model.save('myModel.h5')

    return tf.estimator.EstimatorSpec



def generate_dataset(data_dir):
    """Generates a speech dataset."""
    audio_config = dataset.AudioConfig(
        sample_rate=flags_obj.sample_rate,
        window_ms=flags_obj.window_ms,
        stride_ms=flags_obj.stride_ms,
        normalize=True
    )
    dataset_config = dataset.DatasetConfig(
        audio_config,
        data_dir,
        flags_obj.vocabulary_file,
        flags_obj.sortagrad
    )
    speech_dataset = dataset.DeepSpeechDataset(dataset_config)
    return speech_dataset


def per_device_batch_size(batch_size, num_gpus):
    """Computes the batch size per device for multi-GPU training."""
    if num_gpus <= 1:
        return batch_size

    remainder = batch_size % num_gpus
    if remainder:
        new_batch_size = batch_size - remainder
        raise ValueError(
            f"When running with multiple GPUs, batch size must be a multiple of "
            f"the number of available GPUs. Found {num_gpus} GPUs with a batch "
            f"size of {batch_size}; try --batch_size={new_batch_size} instead."
        )
    return int(batch_size / num_gpus)

def run_deep_speech(_):
  """Run deep speech training and evaluation loop."""
  
  # Set random seed
  tf.random.set_seed(flags_obj.seed)
  
  # Data preprocessing
  logging.info("Data preprocessing...")
  train_speech_dataset = generate_dataset(flags_obj.train_data_dir)
  eval_speech_dataset = generate_dataset(flags_obj.eval_data_dir)
  
  # Get number of label classes
  num_classes = len(train_speech_dataset.speech_labels)
  
  # Set up distribution strategy for multi-GPU training
  num_gpus = flags_core.get_num_gpus(flags_obj)
  distribution_strategy = distribution_utils.get_distribution_strategy(num_gpus=num_gpus)
  run_config = tf.estimator.RunConfig(train_distribute=distribution_strategy)
  
  # Set up estimator
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=flags_obj.model_dir,
      config=run_config,
      params={
          "num_classes": num_classes,
      }
  )
  
  # Set up parameters for benchmark logging
  run_params = {
      "batch_size": flags_obj.batch_size,
      "train_epochs": flags_obj.train_epochs,
      "rnn_hidden_size": flags_obj.rnn_hidden_size,
      "rnn_hidden_layers": flags_obj.rnn_hidden_layers,
      "rnn_type": flags_obj.rnn_type,
      "is_bidirectional": flags_obj.is_bidirectional,
      "use_bias": flags_obj.use_bias
  }
  
  # Get per-device batch size for multi-GPU training
  per_replica_batch_size = per_device_batch_size(flags_obj.batch_size, num_gpus)
  
  # Define input functions for training and evaluation
  def input_fn_train():
      return dataset.input_fn(per_replica_batch_size, train_speech_dataset)
  
  def input_fn_eval():
      return dataset.input_fn(per_replica_batch_size, eval_speech_dataset)
  
  # Set total number of training cycles
  total_training_cycle = (flags_obj.train_epochs // flags_obj.epochs_between_evals)
  
  # Training loop
  for cycle_index in range(total_training_cycle):
      logging.info("Starting training cycle %d/%d", cycle_index + 1, total_training_cycle)
      
      # Perform batch-wise dataset shuffling
      train_speech_dataset.entries = dataset.batch_wise_dataset_shuffle(
          train_speech_dataset.entries,
          cycle_index,
          flags_obj.sortagrad,
          flags_obj.batch_size
      )
      
      # Train the estimator
      estimator.train(input_fn=input_fn_train)
      
      # Evaluate the model
      logging.info("Starting evaluation...")
      eval_results = evaluate_model(
          estimator,
          eval_speech_dataset.speech_labels,
          eval_speech_dataset.entries,
          input_fn_eval
      )
      
      # Log the WER and CER results
      logging.info(
          "Iteration %d: WER = %.2f, CER = %.2f",
          cycle_index + 1,
          eval_results[_WER_KEY],
          eval_results[_CER_KEY]
      )
      
      # Check if evaluation threshold is met
      if model_helpers.past_stop_threshold(eval_results[_WER_KEY], flags_obj.wer_threshold):
          break



def define_deep_speech_flags():
  """Add flags for run_deep_speech."""
  # Add common flags
  flags_core.define_base(
      data_dir=False,  # we use train_data_dir and eval_data_dir instead
      export_dir=True,
      train_epochs=True,
      hooks=True,
      num_gpu=True,
      epochs_between_evals=True
  )
  flags_core.define_performance(
      num_parallel_calls=False,
      inter_op=False,
      intra_op=False,
      synthetic_data=False,
      max_train_steps=False,
      dtype=False
  )
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags_core.set_defaults(
      model_dir="/mnt/c/Users/xiang/Development/trained_model/",
      export_dir="/mnt/c/Users/xiang/Development/deep_speech_saved_model/",
      train_epochs=10,
      batch_size=32,
      hooks="")

  # Deep speech flags
  flags.DEFINE_integer(
      name="seed", default=1,
      help=flags_core.help_wrap("The random seed."))

  flags.DEFINE_string(
      name="train_data_dir",
      default="/tmp/librispeech_data/test-clean/LibriSpeech/test-clean.csv",
      help=flags_core.help_wrap("The csv file path of train dataset."))

  flags.DEFINE_string(
      name="eval_data_dir",
      default="/tmp/librispeech_data/test-clean/LibriSpeech/test-clean.csv",
      help=flags_core.help_wrap("The csv file path of evaluation dataset."))

  flags.DEFINE_bool(
      name="sortagrad", default=True,
      help=flags_core.help_wrap(
          "If true, sort examples by audio length and perform no "
          "batch_wise shuffling for the first epoch."))

  flags.DEFINE_integer(
      name="sample_rate", default=16000,
      help=flags_core.help_wrap("The sample rate for audio."))

  flags.DEFINE_integer(
      name="window_ms", default=20,
      help=flags_core.help_wrap("The frame length for spectrogram."))

  flags.DEFINE_integer(
      name="stride_ms", default=10,
      help=flags_core.help_wrap("The frame step."))

  flags.DEFINE_string(
      name="vocabulary_file", default=_VOCABULARY_FILE,
      help=flags_core.help_wrap("The file path of vocabulary file."))

  # RNN related flags
  flags.DEFINE_integer(
      name="rnn_hidden_size", default=800,
      help=flags_core.help_wrap("The hidden size of RNNs."))

  flags.DEFINE_integer(
      name="rnn_hidden_layers", default=5,
      help=flags_core.help_wrap("The number of RNN layers."))

  flags.DEFINE_bool(
      name="use_bias", default=True,
      help=flags_core.help_wrap("Use bias in the last fully-connected layer"))

  flags.DEFINE_bool(
      name="is_bidirectional", default=True,
      help=flags_core.help_wrap("If rnn unit is bidirectional"))

  flags.DEFINE_enum(
      name="rnn_type", default="gru",
      enum_values=deep_speech_model.SUPPORTED_RNNS.keys(),
      case_sensitive=False,
      help=flags_core.help_wrap("Type of RNN cell."))

  # Training related flags
  flags.DEFINE_float(
      name="learning_rate", default=5e-4,
      help=flags_core.help_wrap("The initial learning rate."))

  # Evaluation metrics threshold
  flags.DEFINE_float(
      name="wer_threshold", default=None,
      help=flags_core.help_wrap(
          "If passed, training will stop when the evaluation metric WER is "
          "greater than or equal to wer_threshold. For libri speech dataset "
          "the desired wer_threshold is 0.23 which is the result achieved by "
          "MLPerf implementation."))


def main(_):
  run_deep_speech(flags_obj)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  define_deep_speech_flags()
  flags_obj = flags.FLAGS
  absl_app.run(main)

