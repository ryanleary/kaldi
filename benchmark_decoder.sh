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

model_path_hash=$(echo $model_path | md5sum | cut -c1-8)
mkdir -p $result_path

# check to see if model needs 8 or 16k data
test_set_suffix="-wav" # assume 16k
sr=$(cat $model_path/conf/mfcc.conf | grep sample-frequency | cut -d= -f2 | awk '{print $1};')
if [[ $sr -eq "8000" ]]; then
  test_set_suffix="-wav8k"
fi
# copy vocabulary locally as lowercase (see below caveat for comment on this)
cat $model_path/words.txt | tr '[:upper:]' '[:lower:]' > $result_path/words.txt

echo "Using the $test_set_suffix version of the dataset..."
# copy the dataset to a local path
  echo "Copying dataset metadata to $local_data..."
  mkdir -p $local_data
  for test_set in test_clean test_other; do
    if [ -d "$local_data/$test_set" ]; then
      echo "Removing existing dataset metadata in $local_data/$test_set"
      rm -rf $local_data/$test_set
    fi
    mkdir -p $local_data/$test_set

    # make a wav.scp manifest that contains the new, local path and the desired
    # sample rate version of the data.
    # NOTE: THIS ASSUMES we have preprocessed the flac audio and generated WAV
    # versions in different sample rates. This is still a manual process and
    # somewhat librispeech specific
    test_set_dash=$(echo $test_set | sed 's/_/-/g')
    cat $librispeech_path/$test_set/wav.scp | awk '{print $1" "$6}' | sed -r "s#(.*) (.*)/$test_set_dash/(.*).flac#\1 $local_data/${test_set_dash}${test_set_suffix}/\3.wav#g" > $local_data/$test_set/wav.scp
    head $local_data/$test_set/wav.scp > $local_data/$test_set/wav_head.scp
    src=$(head -n 1 $librispeech_path/$test_set/wav.scp |  awk '{print $6}' | cut -d '/' -f -5 | sed -r "s#(.*)/$test_set_dash#\1/${test_set_dash}${test_set_suffix}#g")

    # generate reference transcripts (using token ids) for the dataset/model pair
    # for simplicity, we'll force reference transcripts and token list lowercase
    # but this isn't necessarily safe depending on language
    echo "Generating new reference transcripts for model and dataset..."
    cat $librispeech_path/$test_set/text | tr '[:upper:]' '[:lower:]' > $local_data/$test_set/text
    oovtok=$(cat $result_path/words.txt | grep "<unk>" | awk '{print $2}')
    ./egs/wsj/s5/utils/sym2int.pl --map-oov $oovtok -f 2- $result_path/words.txt $local_data/$test_set/text > $local_data/$test_set/text_ints_$model_path_hash &> /dev/null

    # copy the correct data for each test set locally
    if [ -d "$local_data/${test_set_dash}${test_set_suffix}" ]; then
      echo "Converted data already exists locally for $test_set_dash, skipping..."
    else
      echo "Copying converted data locally for $test_set_dash..."
      echo "$src"
      cp -R $src $local_data/
    fi
  done

for decoder in online2-wav-nnet3-cuda online2-wav-nnet3-cpu; do # online2-wav-nnet3-faster; do
  for test_set in test_clean test_other; do
    log_file="$result_path/log.$decoder.$test_set.out"

    threads=8
    if [[ $decoder = *"-cpu" ]]; then
      threads=40
    fi

    # run the target decoder with the current dataset
    echo "Running $decoder decoder on $test_set$trunc [$threads threads]..."
    OMP_NUM_THREADS=$threads ./src/online2bin/$decoder \
      --frames-per-chunk=160 --online=false --do-endpointing=false --frame-subsampling-factor=3 \
      --config="$model_path/conf/online.conf" \
      --beam=15.0 --acoustic-scale=1.0 \
      $model_path/final.mdl \
      $model_path/HCLG.fst \
      "ark:$librispeech_path/$test_set/$spk2utt" \
      "scp:$local_data/$test_set/$wavscp" \
      "ark:|gzip -c > $result_path/lat.$decoder.$test_set.gz" &> $log_file 2>&1

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
      "ark:$local_data/$test_set/text_ints_$model_path_hash" \
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
