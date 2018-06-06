rm -f audio.txt ids.txt

DIR=$1
FILES=`ls -1 $DIR`
i=0;

for file in $FILES; do
  i=`echo $i+1 | bc`
  echo "utt_$i $DIR/$file" >> audio.txt
  echo "utt_$i utt_$i" >> ids.txt
done

