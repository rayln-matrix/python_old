import math
import os
import random
import pandas as pd
import numpy as np
import string



#Write a Python program to calculate the length of a string
test_randint_list=[]
for i in range(50):
    rand_int=random.randint(1,100)
    test_randint_list.append(rand_int)
#print(test_randint_list)

def len_of_list(list_data):
    count = 0
    for content in list_data:
        count +=1
        content = content
    return count


def len_of_string(string_data):
    count = 0
    for char in string_data:
        count += 1
        char = char
    return count

# print(len_of_list(test_randint_list))
# print(len(test_randint_list))

# print(len_of_string('ajfatlkjlkjlkj'))
# print(len('ajfatlkjlkjlkj'))


#Write a Python program to count the number of characters (character frequency) in a string:
#Sample String : google.com'
#Expected Result : {'o': 3, 'g': 2, '.': 1, 'e': 1, 'l': 1, 'm': 1, 'c': 1}

def count_char(string_data):
    #char_set= set(string.ascii_lowercase).union(set(string.ascii_uppercase))
    char_set=set(string.ascii_letters)
    # print(len(char_set))
    # print(len(char_set3))
    char_fequency=dict()
    for char in char_set:
        char_fequency[char]=0
    for content in string_data:
        for char in char_set:
            if content == char:
                char_fequency[char] += 1
    return char_fequency


#sample solution:
def char_frequency(str1):
    dict = {}
    for n in str1:
        keys = dict.keys()
        if n in keys:
            # When started, n is not in keys, therefore skip this process and move to else:
            dict[n] += 1
        else:
            # n is not in keys, and add n to key and assign 1 to it's value
            dict[n] = 1
    return dict

#print(count_char('google.com'))
#print(char_frequency('google.com'))



#Write a Python program to get a string made of the first 2 and the last 2 chars from a given a string.
# If the string length is less than 2, return instead of the empty string
# Test:
# Sample String : 'w3resource'
# Expected Result : 'w3ce'
# Sample String : 'w3'
# Expected Result : 'w3w3'
# Sample String : ' w'
# Expected Result : Empty String


def combined_string(string_data):
    if len(string_data) >= 2:
        return string_data[:2]+string_data[-2:]
    else:
        return 'Emty String'
# print(combined_string('w3resource'))
# print(combined_string('w3'))
# print(combined_string('w'))


#Sample solution:
def string_both_ends(str):
    if len(str) < 2:
        return ''
    return str[0:2] + str[-2:]



# Write a Python program to get a string from a given string where all occurrences of its first char have been changed to '$',
# except the first char itself.
# Sample String : 'restart'
# Expected Result : 'resta$t'

# Caution: String object does not support iten assignment
def changing_content(string_data):
    first = string_data[0]
    string_data=string_data.replace(first,'$')
    string_data=first+string_data[1:]
    return string_data

#print(changing_content('restart'))

#Sample solution:
def change_char(str1):
    char = str1[0]
    str1 = str1.replace(char, '$')
    str1 = char + str1[1:]
    return str1


# Write a Python program to get a single string from two given strings,
# separated by a space and swap the first two characters of each string.
# Sample String : 'abc', 'xyz'
# Expected Result : 'xyc abz'

def combined_swap(str1,str2):
    return str2[:-1]+str1[-1]+' '+str1[:-1]+str2[-1]

#print(combined_swap('abc','xyz'))


#Sample solution:
def chars_mix_up(a, b):
    new_a = b[:2] + a[2:]
    new_b = a[:2] + b[2:]
    return new_a + ' ' + new_b


# Write a Python program to add 'ing' at the end of a given string (length should be at least 3).
# If the given string already ends with 'ing' then add 'ly' instead.
# If the string length of the given string is less than 3, leave it unchanged.
# Sample String : 'abc'
# Expected Result : 'abcing'
# Sample String : 'string'
# Expected Result : 'stringly'


def adding_content(str1):
    if len(str1) < 3:
        return str1
    if str1[-3:]=='ing':
        return str1+'ly'
    return str1+'ing'

# print(adding_content('ab'))
# print(adding_content('abcing'))
# print(adding_content('string'))


#Sample solution:
def add_string(str1):
    length = len(str1)
    if length > 2:
       if str1[-3:] == 'ing':
            str1 += 'ly'
       else:
            str1 += 'ing'
    return str1


# Write a Python program to find the first appearance of the substring 'not' and 'poor' from a given string,
# if 'not' follows the 'poor', replace the whole 'not'...'poor' substring with 'good'.
#  Return the resulting string.
# Sample String : 'The lyrics is not that poor!'
# 'The lyrics is poor!'
# Expected Result : 'The lyrics is good!'
# 'The lyrics is poor!'

# Find method will find the index of first char of the input substring
def content_sub(str1):
    indx_not=str1.find('not')
    indx_poor=str1.find('poor')
    poor_end=indx_poor+4 # +3-->r, +1-->for the slicing
    if indx_not == -1:
        return str1
    if indx_not < indx_poor:
        sub_str=str1[indx_not:poor_end]
        return str1.replace(sub_str,'good')


# print(content_sub('The lyrics is not that poor!'))
# print(content_sub('The lyrics is poor!'))


#Sample solution:
def not_poor(str1):
    snot = str1.find('not')
    spoor = str1.find('poor')
    if spoor > snot and snot>0 and spoor>0:
        str1 = str1.replace(str1[snot:(spoor+4)], 'good')
        return str1
    else:
        return str1


# 8. Write a Python function that takes a list of words and returns the length of the longest one.
def longest(list_str):
    str_len=0
    for s in list_str:
        if len(s) > str_len:
            str_len=len(s)
            long_str=s
    return long_str
#print(longest(["PHP", "Exercises", "Backend"]))


#Sample solution:
def find_longest_word(words_list):
    word_len = []
    for n in words_list:
        word_len.append((len(n), n))
    word_len.sort()
    return word_len[-1][1]

#print(find_longest_word(["PHP", "Exercises", "Backend"]))

# 9.Write a Python program to remove the nth index character from a nonempty string.

def remove_something(str1,idx):
    if len(str1) ==0:
        return str1
    return str1[:idx]+str1[(idx+1):]


# print(remove_something('Python', 0))
# print(remove_something('Python', 3))
# print(remove_something('Python', 5))
# print(remove_something('',0))

#Sample solution:

def remove_char(str, n):
      first_part = str[:n]
      last_part = str[n+1:]
      return first_part + last_part
# print(remove_char('Python', 0))
# print(remove_char('Python', 3))
# print(remove_char('Python', 5))
# print(remove_char('',0))


# 10. Write a Python program to change a given string to a new string where the first and last chars have been exchanged.

def swap_first_last(str1):
    return str1[-1]+str1[1:-1]+str1[0]

# print(swap_first_last('abcd'))
# print(swap_first_last('12345'))

#Sample solution:
def change_sring(str1):
      return str1[-1:] + str1[1:-1] + str1[:1]

# print(change_sring('abcd'))
# print(change_sring('12345'))


# 84. Write a Python program to swap cases of a given string.
# Sample Output:
# pYTHON eXERCISES
# jAVA
# nUMpY

def swap_case(str1):
    str2=''
    for char in str1:
        if char.isupper():
            str2=str2+char.lower()
        if char.islower():
            str2=str2+char.upper()
        if char==' ':
            str2=str2+char
    return str2

# print(swap_case('Python Exercise'))
# print(swap_case('Java'))
# print(swap_case('NumPy'))

#Sample solution:
def swap_case_string(str1):
   result_str = ""   
   for item in str1:
       if item.isupper():
           result_str += item.lower()
       else:
           result_str += item.upper()           
   return result_str
# print(swap_case_string("Python Exercises"))
# print(swap_case_string("Java"))
# print(swap_case_string("NumPy"))

# 83. Write a Python program to print four values decimal, octal, hexadecimal (capitalized), binary in a single line of a given integer. 
# Sample Output:
# Input an integer: 25
# Decimal Octal Hexadecimal (capitalized), Binary
# 25 31 19 11001

## How to effectively convert decimal to octal, hexadecimal and binary?
def int_convert(int1):
    # deci=int1
    # octa=int1/8
    # hexa=int1
    return

def convert_oct(int1):
    if int1 == 8:
        return 10
    if int1 < 8:
        return int1
    div=int1/8
    if div > 8:
        return convert_oct(div)+1


#Sample solution:
# i = int(input("Input an integer: "))
# o = str(oct(i))[2:]
# h = str(hex(i))[2:]
# h = h.upper()
# b = str(bin(i))[2:]
# d = str(i)
# print("Decimal Octal Hexadecimal (capitalized), Binary")
# print(d,'  ',o,' ',h,'                   ',b)


# Write a Python program to wrap a given string into a paragraph of given width.
# Sample Output:
# Input a string: The quick brown fox.
# Input the width of the paragraph: 10
# Result:
# The quick
# brown fox.


# str1=input("Input a string:")
# w_len=int(input("Input the width of the paragraph:"))
# block_numb=len(str1)//w_len
# while block_numb >=1:
#     print(str1[:w_len])
#     str1=str1[w_len:]
#     block_numb -= 1    
    

#Sample solution:
# import textwrap
# s = input("Input a string: ")
# w = int(input("Input the width of the paragraph: ").strip())
# print("Result:")
# print(textwrap.fill(s,w))


# 74. Write a Python program to find the minimum window in a given string which will contain all the characters of another given string. 
# Example 1
# Input : str1 = " PRWSOERIUSFK "
# str2 = " OSU "
# Output: Minimum window is "OERIUS"

def minimum_window(str1,str2):
    
    return




#Sample solution:


# 75. Write a Python program to find smallest window that contains all characters of a given string.
#Sample solution:


# 76. Write a Python program to count number of substrings from a given string of lowercase alphabets with exactly k distinct (given) characters.
#Sample solution:


# 77. Write a Python program to count number of non-empty substrings of a given string.
#Sample solution:


# 78. Write a Python program to count characters at same position in a given string (lower and uppercase characters) as in English alphabet.
#Sample solution:


# 79. Write a Python program to find smallest and largest word in a given string.
#Sample solution:


# 80. Write a Python program to count number of substrings with same first and last characters of a given string. 
# #Sample solution:


# 81. Write a Python program to find the index of a given string at which a given substring starts. If the substring is not found in the given string return 'Not found'.
#Sample solution:



# Write a Python program for binary search. Go to the editor
# Binary Search : In computer science, a binary search or half-interval search algorithm finds the position of a target value within a sorted array. The binary search algorithm can be classified as a dichotomies divide-and-conquer search algorithm and executes in logarithmic time.
# Test Data :
# binary_search([1,2,3,5,8], 6) -> False
# binary_search([1,2,3,5,8], 5) -> True




def biny_search(item_list,item):
	first = 0
	last = len(item_list)-1
	found = False
	while( first<=last and not(found)):
		mid = (first + last)//2
		if item_list[mid] == item :
			found = True
		else:
			if item < item_list[mid]:
				last = mid - 1
			else:
				first = mid + 1	
	return found
	
# print(biny_search([1,2,3,5,8], 6))
# print(biny_search([1,2,3,5,8], 5))



# 
count=2
k1= math.log(1.0001, count)
k2= count*2
# while k2>k1:
#     count += 100
##print(count)
# #

def find_peak_brutal(list_data):
    # input: a list of numbers
    # output: a single numbers or None
    # Definition: a list of three number: a b c --> if b >= a and b >= c then b is a peaK
    # this function find a peak: there can be several peak in a list but this function will return only one
    # Peak is different from grobal maximum
    # This is the brutal way to find a peak.
    
    # Check if the first and last are the peak:
    if len(list_data) <= 1:
        return list_data
    if list_data[0] >= list_data[1]:
        return list_data[0]
    if list_data[-1]  >= list_data[-2]:
        return list_data[-1]
    #base=1
    for i in range(len(list_data)):
        while i > 0 and i < len(list_data):
            if list_data[i] >= list_data[i+1]:
                if list_data[i] >= list_data[i-1]:
                    return list_data[i]
            else:
                break
            #base += 1
    return None


# print(find_peak_brutal([1,3,2,4,5,4]))
# print(find_peak_brutal([3,1,1,1,1,1,0]))
# print(find_peak_brutal([5,4,3,2,1,6]))
# print(find_peak_brutal([1,2,3,4,5,6,7,6]))

# def testing(func):
#     for epco in range(50):
#         rand_lenth=random.randint(1,50)
#         rand_int_list=[]
#         for d in range(rand_lenth):
#             rand_int_list.append(random.randint(0,100))
#         print(rand_int_list)
#         print(func(rand_int_list))

#testing(find_peak_brutal)
#print(rand_int_list)

def find_peak_half(list_data):
    
    return


def create_stack(length_int):
    if length_int ==0:
        return None


# class stacK_pract (self):
#     self.length

#     def isfull():
#         return
    
#     def pop():
#         return

#     def add():
#         return


### ensemble learning
#1. decision stump
#2. addboost
#3. random forest
#4. Cart tree
#5. blending 

### SVM
#4. SVM
#5. kernel ridge regression


### ANN: artificial neural network 
#6. perceptron, MLP, neural network
#7. autoenconder, factorized matrix
#8. k- means, k-nearest neighbor
#9. RBF networl


# test_string_list=['A an appLE', 'B a Ball']
# for s in test_string_list:
#     if s.lower() == 'a an apple':
#         print(True)


##

def I_rate_convert(numb,i_rate):
    # for one year
    retrun_rate=1+i_rate
    month=0
    while month < 12:
        numb=numb*retrun_rate
        month += 1
    return (numb,i_rate)

#(1000,10%), (1000, 8%), (1000,5%), (1000,3%) 
# print(I_rate_convert(1000,0.1))
# print(I_rate_convert(1000,0.08))
# print(I_rate_convert(1000,0.05))
# print(I_rate_convert(1000,0.03))

def expect_p(numb,b_rate):
    r_rate=1+b_rate
    month=0
    while month < 12:
        numb= numb/r_rate
        month +=1
    return (numb,b_rate)

#A:1000, i_rate=10%, b_rate=5% VS B: 1000, i_rate=b_rate=5%

#A:
i_return=I_rate_convert(1000,0.1)[0]
#print(I_rate_convert(1000,0.1))
#print(i_return)

b_price=expect_p(i_return,0.05)[0]
#print(expect_p(i_return,0.05))
#print(b_price)


#B:
i_return_b=I_rate_convert(1000,0.05)[0]
#print(I_rate_convert(1000,0.05))
#print(i_return_b)
#---->5%/month, return 1795.85 

b_price_b=expect_p(i_return_b,0.10)[0]
#print(expect_p(i_return_b,0.10))
#print(b_price_b)
#---->expect rate is 10%/month, return 1795.85. what's the origin price?--->1795.85/(1.1)**12=572.2151
#----> because you put 572.2151 in bank for 10% will return the same 


#Sequential search:

def seq_search(lst1,int1):
    for i in lst1:    # steps: Len(lst1)=n
        if i == int1: # steps: 1
            return True  #steps:1
    return False  # steps:1
## total: n*(1+1)+1=O(n)

#print(seq_search([1,2,3,5,8],6))
#print(seq_search([1,2,3,5,8],5))
#print(seq_search([11,23,58,31,56,77,43,12,65,19],31))

# needed to be sorted:
def bin_search(lst1,int1):
    start=0          #steps=1
    end=len(lst1)-1  #steps=1
    #half=(start+end)//2 #steps=1 (S=1)
    found=False
    while (start<=end) and not(found):
        half=(start+end)//2
        if lst1[half] == int1:
            found = True
        if lst1[half] > int1:
            end = half -1
        else:
            start= half+1
    return found
## total: log2(n)


#print(bin_search([1,2,3,5,8],6))
#print(bin_search([1,2,3,5,8],5))
#print(bin_search([1,3,4,5,6,7,8,9,11,12,15,46,59,100],59))
#print(seq_search([11,23,58,31,56,77,43,12,65,19],31))

def brutal_sort(list1):
    ## 1. find the gloabal minimum, switch it with the left most item
    ## 2. slice the list and do it again
    if len(list1) ==1:       # 1
        return [list1[0]]      # 1
    left_most=list1[0]       # 1
    mini=left_most           # 1
    idx_of_min=0             # 1
    for i in range(len(list1)): # n+2
        if list1[i] <= mini:    # 1
            mini = list1[i]     # 1
            idx_of_min = i      # 1
    left_most=mini  # 1
    list1.remove(list1[idx_of_min]) # 1
    #above: O=k*n + k
    return [left_most]+brutal_sort(list1) # 1 + O(n-1)

#total= n + n-1 +....1 ---> (n+1)n/2= k(n**2+n)~ n**2
#print(brutal_sort([11,23,58,31,56,77,43,12,65,19]))


### A class to generate list of random integer
class Randint_List():
    
    def __init__(self,length_of_list):
        self.length_of_list = length_of_list
    
    def generate(self):
        length=self.length_of_list
        if length == 0:
            return "Length 0 is not possible"
        randit_list=[]
        for i in range(length):
            randit_list.append(random.randint(0,100))
            i =i 
        return randit_list

### Check for brutal sort: it works
# for i in range(1,30):
#     test_list_object=Randint_List(i)
#     test_list=test_list_object.generate()
#     print(test_list)
#     print(brutal_sort(test_list))


## Bubble sort (pair_exchange), selection sort, insertion sort, shell sort, merge sort, quick sort, counting sort, Bitonic sort, Bogosort sort, Gnome sort, 
## Cocktail shaker sort, Comb sort, cycle sort, heap sort, Pancake sort, Radix sort, Time sort, Topological sort, Tree sort 

def bubble(list1):
    # pair-wise compare and exchange
    if len(list1) == 1:    #1
        return list1       #1
    #mini=list1[0]
    #middle=0
    for i in range(len(list1)):       #n
        if i+1 <= len(list1)-1:       #1+1+1+1
            if list1[i] > list1[i+1]: #1+1+1
                middle = list1[i]     #1+1
                list1[i] = list1[i+1] #1+1+1
                list1[i+1] = middle   #1+1+1
                #list1[i] = middle
    #print(list)
    # O=k*n+k    
    return bubble(list1[:-1])+[list1[-1]]  #O(n-1)
    # total: O(n**2)

    
# testing: 1st-->wrong  2nd ---> works: problem: need to recursive do the sub list and concat
#for i in range(1,50):
#    testing=Randint_List(i).generate()
#    print(testing)
#    print(bubble(testing))
#    print(brutal_sort(testing))


#insert sort:
def  insert_sort(list1):
    #insert a item to a sorted list
    return


#test10=[1,2,3,4,5,6,7]
#print(test10[:-5:-1])
#pair=('Dog','n')
#print('word%s.tag%s.01'%pair[0])---> no needed



count = 0
# while count < 100:
#     k1 = random.randint(0,100)
#     k2 = random.randint(100,1000)
#     k3 = random.randint(1000,2000)
#     print(count, [k1,k2,k3])
#     count += 1



class Decstump():
    
    def __init__(self,threshold, learning_rate):
        self.threshold=threshold
        self.learning_rate=learning_rate

    def find_weight(self,data_array):
        return

    def predict(self,data_array):
        threshold=self.threshold
        sum=0
        #data_numb=len(data_array)
        for item in data_array:
            sum = sum + item
        if sum > threshold:
            return 1
        else:
            return 0
    
    def error_estimate(self, predicted, true):
        # 1/0 error is used
        return