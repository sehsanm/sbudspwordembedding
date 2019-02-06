#util/lcs.py
m = len(s1)
n = len(s2)
c = [[0] * (n + 1) for i in range(m + 1)]
bt = [[0] * (n + 1) for i in range(m + 1)]

for i in range(1, m):
    c[i][0] = 0
for j in range(1, n):
    c[0][j] = 0
for i in range(1, m + 1):
    for j in range(1, n + 1):
        c[i][j] = c[i][j - 1]
        bt[i][j] = -1
        if (c[i][j] < c[i - 1][j]):
            c[i][j] = c[i - 1][j]
            bt[i][j] = 1
        elif s1[i - 1] == s2[j - 1] and c[i][j] < c[i - 1][j - 1] + 1:
            c[i][j] = c[i - 1][j - 1] + 1
            bt[i][j] = 0
matched_tuples = []
ind_i = m
ind_j = n

while ind_i > 0 and ind_j > 0:
    if (bt[ind_i][ind_j] == 0):
        matched_tuples.append((ind_i - 1, ind_j - 1))
        ind_i = ind_i - 1
        ind_j = ind_j - 1
    elif bt[ind_i][ind_j] == 1:
        ind_i = ind_i - 1
    else:
        ind_j = ind_j - 1

return list(reversed(matched_tuples))

