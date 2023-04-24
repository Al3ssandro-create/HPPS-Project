../gmm_um 16 ../data/mydata.txt ../output/new_out_check.txt 8
diff ../output/new_out_check.txt.summary ../output/old_out_check.txt.summary 
diff ../output/new_out_check.txt.results ../output/old_out_check.txt.results 
rm ../output/new_out_check.*