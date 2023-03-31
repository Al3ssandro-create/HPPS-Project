../gmm 16 ../data/mydata.txt ../output/out_check.txt 8 | grep "Final rissanen Score was:" > ../output/tmp1.txt
../gmm 16 ../data/mydata.txt ../output/out_check.txt 8 | grep "Final rissanen Score was:" > ../output/tmp2.txt
diff ../output/tmp1.txt ../output/tmp2.txt
rm ../output/tmp1.txt
rm ../output/tmp2.txt
rm ../output/out_check.*