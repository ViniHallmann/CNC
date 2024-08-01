#Algoritmo 1/3 Simpson
simp13 = lambda h, f0, f1, f2: 2*h*(f0 + 4*f1 + f2)/6

simp38 = lambda h, f0, f1, f2, f3: 3*h*(f0 + 3*(f1+f2) + f3)/8

trap = lambda h, f0, f1: h*(f0 + f1)/2

def simp13m(h,n,f):
    sum = f[0]
    for i in range( 1, n-1,2 ):
        sum += 4*f[i] + 2*f[i+1]
    sum += 4*f[n-1] + f[n]
    return h*sum/3

def simpInt(a,b,n,f):
    h = (b-a)/n
    
    if n == 1:
        sum = trap(h,f[n-1],f[n])
    else:
        m = n
        odd = n / 2 - int(n / 2)
        if odd > 0 and n > 1:
            sum += simp38(h,f[n-3],f[n-2],f[n-1],f[n])
            m = n - 3
    
    if m > 1:
        sum += simp13m(h,m,f)
    
    return sum