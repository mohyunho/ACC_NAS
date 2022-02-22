# def vap(x, a, b, c):
#     """ Vapor pressure model """
#     return np.exp(a+b/x+c*np.log(x))

def pow3(x, c, a, alpha):
    return  c - a * x**(-alpha)

# def loglog_linear(x, a, b):
#     x = np.log(x)
#     return np.log(a*x + b)


def loglog_linear(x, a, b, c):
    # x = x+1
    x = np.log(x)
    return -1*np.log(np.abs(a*x - b))+c


# def dr_hill(x, alpha, theta, eta, kappa):
#     return alpha + (theta*(x**eta)) / (kappa**eta + x**eta)


# def log_power(x, a, b, c):
#     #logistic power
#     return -1*a/(10.+ np.abs(x/np.exp(b))**c)

# def pow4(x, c, a, b, alpha):
#     return c - (a*x+b)**-alpha


def mmf(x, alpha, beta, kappa, delta):
    return alpha - (alpha - beta) / (1. + np.abs(kappa * x)**delta)


# def exponential_fit(x, a, b, c):
#     return a*np.exp(-b*x) + c

# def exp3(x, c, a, b):
#     return -c + np.exp(-a*x+b)


# def exp4(x, c, a, b, alpha):
#     return -c + np.exp(-a*(x**alpha)+b)

def janoschek(x, a, beta, k, delta):
    return a - (a - beta) * np.exp(-k*x**delta)

def weibull(x, alpha, beta, kappa, delta):
    x = 1 + x
    return alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)

def ilog2(x, c, a, b):
    x = 1 + x
    assert(np.all(x>1))
    return -c + a / np.log(b*x)

# def dr_hill_zero_background(x, theta, eta, kappa):
#     return (theta* x**eta) / (kappa**eta + x**eta)

# def logx_linear(x, a, b):
#     x = np.log(x)
#     return a*x + b

# def exp3(x, c, a, b):
#     return c - np.exp(-a*x+b)

# def pow2(x, a, alpha):
#     return a * x**(-alpha)

# def sat_growth(x, a, b):
#     return a * x / (b + x)