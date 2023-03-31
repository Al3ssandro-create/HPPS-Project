../gmm 16 ../data/mydata.txt ../output/new_out_check.txt 8 | grep "Final rissanen Score was:" > ../output/new_res.txt
diff ../output/new_out_check.txt.summary ../output/old_out_check.txt.summary 
diff ../output/new_out_check.txt.results ../output/old_out_check.txt.results 
rm ../output/new_res.txt
rm ../output/new_out_check.*