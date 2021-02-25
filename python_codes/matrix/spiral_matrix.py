"""
Date : 25th Feb 2021
Author : Shubham Naik


Print matrix in spiral order
Input:
 
[  1   2   3   4  5 ]
[ 16  17  18  19  6 ]
[ 15  24  25  20  7 ]
[ 14  23  22  21  8 ]
[ 13  12  11  10  9 ]
 
Output:
 
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 

"""

import numpy as np 

m1 = np.ones((4,5))

m2 = np.array([[1,2,3,4,5],
                [16,17,18,19,6],
                [15,24,25,20,7],
                [14,23,22,21,8],
                [13,12,11,10,9]])


while (m2.shape[0] != 0 and m2.shape[1] != 0):
    try:
        print(' '.join(str(e) for e in list(m2[0])), end=' ')              # print first row
        m2 = np.delete(m2, 0, axis=0)                                      # delete first row

        print(' '.join(str(e) for e in list(m2[:,-1])), end=' ')           # print last col
        m2 = np.delete(m2, m2.shape[1]-1, axis=1)                          # delete last col

        print(' '.join(str(e) for e in list(m2[-1,:][::-1])), end=' ')     # print last row
        m2 = np.delete(m2, m2.shape[0]-1, axis=0)                          # delete last col

        print(' '.join(str(e) for e in list(m2[:,0][::-1])), end=' ')      # print first col
        m2 = np.delete(m2, 0, axis=1)                                      # delete first col
    except Exception as e:
        # print('Done')
        pass
