#!/bin/bash

if [ $# -lt 4 ]; then
  echo "Usage: $0 <model_path> <librispeech_path> <result_path> <local_data_path> [truncated]"
  exit 1
fi

model_path=$1
librispeech_path=$2
result_path=$3
local_data=$4

spk2utt="spk2utt"
wavscp="wav.scp"
trunc=""
if [ -n "$5" ]; then
  spk2utt="spk2utt_head"
  wavscp="wav_head.scp"
  trunc=" (truncated)"
fi

mkdir -p $result_path

# copy the dataset to a local path
if [ -d "$local_data" ]; then
  echo "Local directory already exists, skipping..."
else
  echo "Copying data to $local_data..."
  mkdir -p $local_data
  for test_set in test_clean test_other; do
    mkdir -p $local_data/$test_set
    test_set_dash=$(echo $test_set | sed 's/_/-/g')
    cat $librispeech_path/$test_set/wav.scp | awk '{print $1" "$6}' | sed -r "s#(.*) (.*)/$test_set_dash/(.*).flac#\1 $local_data/${test_set_dash}-wav8k/\3.wav#g" > $local_data/$test_set/wav.scp
    head $local_data/$test_set/wav.scp > $local_data/$test_set/wav_head.scp
  done
  cp -R $librispeech_path/LibriSpeech/test-*-wav8k $local_data/
fi

for decoder in online2-wav-nnet3-cuda online2-wav-nnet3-cpu; do # online2-wav-nnet3-faster; do
  for test_set in test_clean test_other; do
    log_file="$result_path/log.$decoder.$test_set.out"

    threads=4
    if [[ $decoder = *"-cpu" ]]; then
      threads=40
    fi

    # run the target decoder with the current dataset
    echo "Running $decoder decoder on $test_set$trunc [$threads threads]..."
    OMP_NUM_THREADS=$threads ./src/online2bin/$decoder \
      --frames-per-chunk=160 --online=false --do-endpointing=false --frame-subsampling-factor=3 \
      --config="$model_path/online.conf" \
      --beam=15.0 --acoustic-scale=1.0 \
      $model_path/final.mdl \
      $model_path/HCLG.fst \
      "ark:$librispeech_path/$test_set/$spk2utt" \
      "scp:$local_data/$test_set/$wavscp" \
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
    expected_sentences=$(cat $local_data/$test_set/$wavscp | wc -l)
    actual_sentences=$(echo $scored | awk '{print $2}')
    echo "  Expected: $expected_sentences, Actual: $actual_sentences"
    if [ $expected_sentences -ne $actual_sentences ]; then
      echo "  Error: did not return expected number of utterances. Check $log_file"
    else
      echo "  Decoding completed successfully."
    fi
  done
done
