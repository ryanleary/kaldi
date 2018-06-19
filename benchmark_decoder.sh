#!/bin/bash

if [ $# -lt 3 ]; then
  echo "Usage: $0 <model_path> <librispeech_path> <result_path> [truncated]"
  exit 1
fi

model_path=$1
librispeech_path=$2
result_path=$3

spk2utt="spk2utt"
wavscp="wav_8k.scp"
trunc=""
if [ -n "$4" ]; then
  spk2utt="spk2utt_head"
  wavscp="wav_8k_head.scp"
  trunc=" (truncated)"
fi

mkdir -p $result_path

for decoder in online2-wav-nnet3-cuda; do # online2-wav-nnet3-faster; do
  for test_set in test_clean test_other; do
    log_file="$result_path/log.$decoder.$test_set.out"

    # run the target decoder with the current dataset
    echo "Running $decoder decoder on $test_set$trunc..."
    ./src/online2bin/$decoder \
      --frames-per-chunk=160 --online=false --do-endpointing=false --frame-subsampling-factor=3 \
      --config="$model_path/online.conf" \
      --beam=15.0 --acoustic-scale=1.0 \
      $model_path/final.mdl \
      $model_path/HCLG.fst \
      "ark:$librispeech_path/$test_set/$spk2utt" \
      "scp:$librispeech_path/$test_set/$wavscp" \
      "ark:|gzip -c > $result_path/lat.$decoder.$test_set.gz" &> $log_file

    if [ $? -ne 0 ]; then
      echo "  Error encountered while decoding. Check $log_file"
      continue
    fi

    # output processing speed from debug log
    rtf=$(cat $log_file | grep RealTimeX | cut -d' ' -f 3-)
    echo "  $rtf"

    # convert lattice to transcript
    ./src/latbin/lattice-best-path \
      "ark:gunzip -c $result_path/lat.$decoder.$test_set.gz |"\
      "ark,t:|gzip -c > $result_path/trans.$decoder.$test_set.gz" >>$log_file 2>&1 

    # calculate wer
    ./src/bin/compute-wer --mode=present \
      "ark:$librispeech_path/$test_set/text_ints" \
      "ark:gunzip -c $result_path/trans.$decoder.$test_set.gz |" >>$log_file 2>&1

    # output accuracy metrics
    wer=$(cat $log_file | grep "%WER")
    ser=$(cat $log_file | grep "%SER")
    scored=$(cat $log_file | grep "Scored")
    echo "  $wer"
    echo "  $ser"
    echo "  $scored"

    # ensure all expected utterances were processed
    expected_sentences=$(cat $librispeech_path/$test_set/$wavscp | wc -l)
    actual_sentences=$(echo $scored | awk '{print $2}')
    echo "  Expected: $expected_sentences, Actual: $actual_sentences"
    if [ $expected_sentences -ne $actual_sentences ]; then
      echo "  Error: did not return expected number of utterances. Check $log_file"
    else
      echo "  Decoding completed successfully."
    fi
  done
done
