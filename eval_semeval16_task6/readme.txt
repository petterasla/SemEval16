##########################################
#This simple evaluation package is created for: 
#
#Semeval-2016 Task 6: Detecting Stance in Tweets
#http://alt.qcri.org/semeval2016/task6/
#
#which is organized by 
#Saif M Mohammad
#Svetlana Kiritchenko
#Parinaz Sobhani
#Xiaodan Zhu
#Colin Cherry
#
#The evaluation script is created by Xiaodan Zhu. If
#you have any questions, please drop a line to
#zhu2048@gmail.com
##########################################


eval.pl
A perl script for evaluation. 
Type "perl eval.pl -u" for usage,
or "perl eval.pl gold_toy.txt guess_toy.txt" as an example.

gold_toy.txt
This is a toy gold-standard file.

guess_toy.txt
This is a toy predication file that contains your prediction of the stance of each tweet.

guess_toy_unix.txt
Same as guess_toy.txt, but with a unix/linux "newline" at the end of each line.
So if you generate your prediction file in unix/linus, the evaluation scripts should
work as well.

readme.txt
