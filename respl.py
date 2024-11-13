'''
Author: xinao_seven_
Date: 2024-11-09 10:14:56
LastEditTime: 2024-11-12 14:58:45
LastEditors: xinao_seven_
Description: 
Encoding: utf-8
FilePath: /ML/respl.py

'''
from typing import List
class Solution:
    def countPairs(self, nums: List[int]) -> int:
        hash = {}
        
        def getdigit(num):
            res = []
            while num>0:
                res.append(num%10)
                num= num//10
            res.reverse()
            return res
        res = 0
        def getnum(digti,i,j):
            digti[i],digti[j] = digti[j],digti[i]
            res = 0
            for k in range(len(digti)):
                res = res*10+digti[k]
            digti[i],digti[j] = digti[j],digti[i]
            return res

        for num in nums:
            digti = getdigit(num)
            for i in range(len(digti)):
                for j in range(i+1,len(digti)):
                    
                    new_num = getnum(digti,i,j)
                    
                    if new_num in hash:
                        
                        hash[new_num] += 1
                    else:
                        hash[new_num] = 1
        log = []
        for num in nums:
            if num in hash:
                res += hash[num]
                log.append([num,hash[num]])
        print(log)
        return res
s1 = Solution()
s1.countPairs([3,12,30,17,21])