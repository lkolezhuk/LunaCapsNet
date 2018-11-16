def binaryGap(N):
    x = str(bin(N)).strip('0b')
    max_len = 0
    curr_len = 0
    for i in range(len(x)):
        val = x[i]
        if val != '1':
            curr_len += 1
        else:
            if curr_len > max_len:
                max_len = curr_len
                curr_len = 0
            else:
                curr_len = 0
    return max_len

from collections import  deque
def rotateArray(A, K):
    if len(A) == 0:
        return A

    K = K % len(A)
    return A[-K:] + A[:-K]
def findUnpaired(A):
    unique = 0
    for i in range(len(A)):
        unique ^=  A[i]
    return unique

def splitArrWithMinDiff(A):
    sub_a = A[0]
    sub_b = sum(A[1:])
    min_dif = abs(sub_a - sub_b)
    for i in range(1, len(A)-1):
        sub_a += A[i]
        sub_b -= A[i]
        dif = abs(sub_a - sub_b)

        if dif < min_dif:
            min_dif = dif

    return min_dif

def findMisingElement(A):

    exp_sum = sum(range(1, len(A)+2))
    got_sum = sum(A)
    return abs(exp_sum - got_sum)

def frogLeaves(X, A):
    locations = [0] * X
    leaves_left = X
    for i in range(len(A)):
        if(locations[A[i]-1] == 0):
            locations[A[i]-1] = 1
            leaves_left -= 1
            if leaves_left == 0:
                return i


def smallestMissingInt(A):

    int_occurances = [False] * (max(A)+1)
    for i in range(0, len(A)):
        if(A[i] > 0):
            int_occurances[A[i]] = True

    for i in range(1, len(int_occurances)):
        if(int_occurances[i] == False):
            return i
        if((i == len(int_occurances)-1)):
            return i+1
    return 1

def isPermutation(A):
    if(len(A)!=max(A)):
        return 0
    int_occurances = [False] * (max(A)+1)
    steps_to_permute = max(A)
    for i in range(0, len(A)):
        if(int_occurances[A[i]] == True):
            return 0

        int_occurances[A[i]] = True
        steps_to_permute -= 1
    if steps_to_permute == 0:
        return 1
    return 0


def counters(N, A):
        result = [0] * N  # The list to be returned
        max_counter = 0  # The used value in previous max_counter command
        current_max = 0  # The current maximum value of any counter

        for command in A:
            if 1 <= command <= N:
                # increase(X) command
                if max_counter > result[command - 1]:
                    # lazy write
                    result[command - 1] = max_counter
                result[command - 1] += 1
                if current_max < result[command - 1]:
                    current_max = result[command - 1]
            else:
                # max_counter command
                # just record the current maximum value for later write
                max_counter = current_max

        for index in range(0, N):
            if result[index] < max_counter:
                # This element has never been used/updated after previous
                #     max_counter command
                result[index] = max_counter

        return result
def divisibleInt(A, B, K):
    import math
    count = math.ceil((B - A) / K)
    if (A % K == 0 | B % K == 0):
        count +=1
    return count
def passingCarsCombinations(A):
    # initialize pairs to zero
    pairs = 0
    # count the numbers of zero discovered while traversing 'A'
    # for each successive '1' in the list, number of pairs will
    # be incremented by the number of zeros discovered before that '1'
    zero_count = 0
    # traverse through the list 'A'
    for i in range(0, len(A)):
        if A[i] == 0:
            # counting the number of zeros discovered
            zero_count += 1
        elif A[i] == 1:
            # if '1' is discovered, then number of pairs is incremented
            # by the number of '0's discovered before that '1'
            pairs += zero_count
            # if pairs is greater than 1 billion, return -1
            if pairs > 1000000000:
                return -1
    # return number of pairs
    return pairs


def triangle(A):
    A_len = len(A)
    if A_len < 3:
        # N is an integer within the range [0..1,000,000]
        # if the list is too short, it is impossible to
        # find out a triangular.
        return 0

    A.sort()

    for index in range(0, A_len - 2):
        if A[index] + A[index + 1] > A[index + 2]:
            return 1
        # The list is sorted, so A[index+i] >= A[index+2]
        # where i>2. If A[index]+A[index+1] <= A[index+2],
        # then A[index]+A[index+1] <= A[index+i], where
        # i>=2. So there is no element in A[index+2:] that
        # could be combined with A[index] and A[index+1]
        # to be a triangular.

    # No triangular is found
    return 0

def numberofunique_elem(A):
    return len(list(set(A)))

def max_product_of_three(A):
    A.sort()
    return max(A[0]*A[1]*A[-1], A[-1]*A[-2]*A[-3])

def brackets(S):
    if len(S) % 2 != 0:
        return 0

    matched = {"]": "[", "}": "{", ")": "("}
    to_push = ["[", "{", "("]
    stack = []

    for element in S:
        if element in to_push:
            stack.append(element)
        else:
            if len(stack) == 0:
                return 0
            elif matched[element] != stack.pop():
                return 0

    if len(stack) == 0:
        return 1
    else:
        return 0
def brackets(S):
    if not S:
        return 1
    if len(S) % 2 != 0:
        return 0

    dict = {"]":"[", "}":"{", ")":"("}
    open_br = ["[", "{", "("]
    stack = deque()

    for s in S:
        if s in open_br:
            stack.append(s)
        else:
            if len(stack) == 0:
                return 0
            elif dict[s] != stack.pop():
                return 0
    if len(stack) == 0:
        return 1
    return 0

def eating_fish(A, B):
    alive_count = 0  # The number of fish that will stay alive
    downstream = []  # To record the fishs flowing downstream
    downstream_count = 0  # To record the number of elements in downstream

    for index in range(len(A)):
        # Compute for each fish
        if B[index] == 1:
            # This fish is flowing downstream. It would
            # NEVER meet the previous fishs. But possibly
            # it has to fight with the downstream fishs.
            downstream.append(A[index])
            downstream_count += 1
        else:
            # This fish is flowing upstream. It would either
            #    eat ALL the previous downstream-flow fishs,
            #    and stay alive.
            # OR
            #    be eaten by ONE of the previous downstream-
            #    flow fishs, which is bigger, and died.
            while downstream_count != 0:
                # It has to fight with each previous living
                # fish, with nearest first.
                if downstream[-1] < A[index]:
                    # Win and to continue the next fight
                    downstream_count -= 1
                    downstream.pop()
                else:
                    # Lose and die
                    break
            else:
                # This upstream-flow fish eat all the previous
                # downstream-flow fishs. Win and stay alive.
                alive_count += 1

    # Currently, all the downstream-flow fishs in stack
    # downstream will not meet with any fish. They will
    # stay alive.
    alive_count += len(downstream)

    return alive_count

def arr_dominator(A):

    len_a = len(A)
    if(len_a == 0): return 0
    cand = A[0]

    count = 0
    for i in range(1, len_a):
        if(A[i] == cand):
            count += 1
        else:
            cand = A[i]
            count = 0
def golden_max_slice(A):
     max_ending = max_slice = A[0]
     for a in A[1:]:
         max_ending = max(0, max_ending + a)
         max_slice = max(max_slice, max_ending)
     return max_slice

def max_profit(A):
        max_end = 0
        max_slice = 0
        for i in range(1, len(A)):
            max_end = max(0, max_end + A[i] - A[i - 1])
            max_slice = max(max_slice, max_end)
        return max_slice

def primality(n):
 i = 2
 while (i * i <= n):
     if (n % i == 0):
         return False
     i += 1
 return True

def divisors(n):
    i = 1
    result = 0
    while (i * i < n):
        if (n % i == 0):
            result += 2
        i += 1
    if (i * i == n):
        result += 1
    return result

def min_rect_perimeter(N):
    import math
    i = 2
    min_perm = 2*(N+1)
    while (i <= round(math.sqrt(N))):
        if(N % i == 0):
            perm = 2*(N/i + i)
            min_perm = min(min_perm, perm)
        i += 1
    return int(round(min_perm))

def gcd(a, b): #gratest common divisor
    if a == b:
        return a
    if a > b:
        gcd(a - b, b)
    else:
        gcd(a, b - a)

def chocolateCircle(N, M):
    lcm = N * M / gcd(N, M) # Least common multiple
    return lcm / M


def solution (AX, AY, BX, BY):

    if(AX == BX):
        if(AY > BY):
            return (BX-1, BY)
        else:
            return (BX+1, BY)
    if(AY == BY):
        if(AY > BY):
            return(BX, BY+1)
        else:
            return (BX, BY-1)
    a = float(BY - AY) / (BX - AX)

    b = -float(AX *(BY - AY)) / (BX - AX) + AY

    bp=BY + float(BX)/a
    ap = -1/a

    move =1
    if(1 % abs(a) == 0):
        move = 1
    else:
        move = gcd(1,abs(a))
    if(AY > BY):
        return (BX-move, ap*(BX-move) + bp)
    else:
        return (BX+move, ap*(BX+move) + bp)


if __name__ == "__main__":
    x = solution(3, 1, 4, 1)
    x = solution(-1, -1, 5, 2)
    x = solution(-1,1,1,2)
    x = solution(-1,-1, 1,4)
     # x = solution(5)
    # x = solution(100004)
    #   x = rotateArray([3, 8, 9, 7, 6], 3)
    #   x = splitArrWithMinDiff([3, 1, 2, 4, 3])
    # x = findMisingElement([2, 3, 1, 5])
    # x = frogLeaves(5, [1, 3, 1, 4, 2, 3, 5, 4])
    # x = smallestMissingInt([1, 2, 3] )
    # x = isPermutation([4, 1, 3, 2])
    # x=counters(5, [3, 4, 4, 6, 1, 4, 4])
    # x = divisibleInt(10,20,3)
    # x = passingCarsCombinations([0, 1, 0, 1, 1] )
    # x = passingCarsCombinations([0, 1, 0, 1, 1, 1])
    # x = brackets("{[()()]}")
    # x = eating_fish([4, 3, 2, 1, 5], [0, 1, 0, 0, 0])
    # x = arr_dominator([3, 4, 3, 2, 3, -1, 3, 3])
    # x= max_profit(  [23171, 21011, 21123, 21366, 21013, 21367] )
    x = min_rect_perimeter(36)
    pass



