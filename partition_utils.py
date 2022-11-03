from numba import jit
import numpy as np
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@jit
def all_partitions(n):
  hold = []
  p = [0] * n  # An array to store a partition
  k = 0    # Index of last element in a partition
  p[k] = n   # Initialize first partition
        # as number itself

  # This loop first prints current partition,
  # then generates next partition.The loop
  # stops when the current partition has all 1s
  while True:
    
      # print current partition
      temp = []
      for i in range(0,k+1):
        temp.append(p[i])
      hold.append(temp)

      # Generate next partition

      # Find the rightmost non-one value in p[].
      # Also, update the rem_val so that we know
      # how much value can be accommodated
      rem_val = 0
      while k >= 0 and p[k] == 1:
        rem_val += p[k]
        k -= 1

      # if k < 0, all the values are 1 so
      # there are no more partitions
      if k < 0:
        print()
        return hold

      # Decrease the p[k] found above
      # and adjust the rem_val
      p[k] -= 1
      rem_val += 1

      # If rem_val is more, then the sorted
      # order is violated. Divide rem_val in
      # different values of size p[k] and copy
      # these values at different positions after p[k]
      while rem_val > p[k]:
        p[k + 1] = p[k]
        rem_val = rem_val - p[k]
        k += 1

      # Copy rem_val to next position
      # and increment position
      p[k + 1] = rem_val
      k += 1



import more_itertools as mit
def all_set_partitions(n):
  lst = list(range(1,n+1))
  return [part for k in range(1, len(lst) + 1) for part in mit.set_partitions(lst, k)]


def binary_words(n,k):
  """
  will return all binary words of 0's and 1's length n with k 1's
  """
  if n<0 or k<0 or k>n:
    return []
  if n==0:
    return [[]]
  B1 = binary_words(n-1,k-1)
  B2 = binary_words(n-1,k)
  R = []
  for b in B1:
    R += [b+[1]]
  for b in B2:
    R += [b+[0]]
  return R


def binary_matricies(pi):
  """
  Given an integer partition pi this will return all len(pi)xN matricies
  row i will have a row sum of pi[i]
  This over produces the matrices we need to count when going from the m-basis to the e-basis
  """
  if len(pi)==1:
    B = binary_words(N,pi[0])
    return [[w] for w in B]
  m = len(pi)
  S = binary_matricies(pi[0:m-1])
  row = binary_matricies([pi[m-1]])
  R = []
  for s in S:
    for r in row:
      R += [s+r]
  return R



def row_sum(M):
  """
  Given a matrix as a list of lists we return the rows sums as a composition
  """
  return [sum(row) for row in M]
def column_sum(M):
  """
  Given a matrix as a list of lists we return the column sums as a composition
  """
  c = []
  for i in range(len(M[0])):
    d = 0
    for j in range(len(M)):
      d += M[j][i]
    c += [d]
  return c



def m_to_e_matrix(a,mu):
  """
  a and mu are integer partitions
  This counts the number of 0-1 matrices with no zero rows or columns
  the row sum is a
  the column sum is mu
  """
  #print(a, mu, "checking term")
  matricies = binary_matricies(a)
  count = 0
  for M in matricies:
    #print(M, "matrix")
    c = column_sum(M)
    #print (c, "column sum")
    is_good = True
    for i in range(len(mu)):
      if not c[i]==mu[i]:
        #print("doesn't match mu")
        is_good = False
    if is_good == True:
      count += 1
  return count




@jit()
def makeMatrix(N):
  #This cell has permanant values that will be used in future methods to save on run time
  PARTITIONS = all_partitions(N) #the collection of integer partitions
  SET_PARTITIONS = all_set_partitions(N)

  #This cell has permanant values that will be used in future methods to save on run time

  M_TO_E_MATRIX = {partition_to_string(pi):{partition_to_string(mu):0 for mu in PARTITIONS} for pi in PARTITIONS}
  #print(M_TO_E_MATRIX)
  for pi in PARTITIONS:
    for mu in PARTITIONS:
      strpi = partition_to_string(pi)
      strmu = partition_to_string(mu)
      #print(m_to_e_matrix(pi,mu))
      (M_TO_E_MATRIX[strpi])[strmu] = m_to_e_matrix(pi,mu)

    return M_TO_E_MATRIX, PARTITIONS, SET_PARTITIONS