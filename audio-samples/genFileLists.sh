rm -f audio.txt ids.txt

DIR=$1
FILES=`ls -1 $DIR/*.wav`
i=0;

for file in $FILES; do
  i=`echo $i+1 | bc`
  echo "utt_$i $file" >> audio.txt
  echo "utt_$i utt_$i" >> ids.txt
done

