../gmm 16 ../data/mydata.txt ../output/new_out_check.txt 8 | grep "Final rissanen Score was:" > ../output/new_res.txt
diff ../output/old_res.txt ../output/new_res.txt
rm ../output/new_res.txt
rm ../output/new_out_check.*