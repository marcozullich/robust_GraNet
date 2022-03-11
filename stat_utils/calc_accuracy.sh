arr=$(cat out*  | grep "Test |" | cut -c 13-18)

tot=0; l=0;
for i in ${arr[@]}; do
	tot=$(python -c "print($tot + $i)")
	l=$((l+1))
done
echo Found $l observations. Average accuracy:
echo $(python -c "print($tot/$l)")
