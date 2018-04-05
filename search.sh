echo "use gpu with multiple processes";
for((i=0;i<=7;i++))
do
    {
     echo "use gpu" +$i ; 
     echo CUDA_VISIBLE_DEVICES=$i python parameter_search.py --gpu $i; 
     CUDA_VISIBLE_DEVICES=$i python parameter_search.py --gpu $i; 
     
     }&
done
wait