import numpy as np

# Constructing a Binomial Tree
# In the following code snippet, you have the code for a general function to simulate the underlying stock price for
# some inputs: initial stock price (S_ini), time-horizon ( 𝑇 ), upward ( 𝑢 ) and downward ( 𝑑 ) movements, and number of
# steps (N).

def binomial_tree(S_ini, T, u, d, N):
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return S

# Stock = binomial_tree(100, 1, 1.2, 0.8, 2)
# print(Stock)

# Extending the Tree with Call Option Payoffs
# Next, let's extend the previous function by adding another variable that computes the payoffs associated with a Call
# Option of certain characteristics. Note that we are focusing on a European Call Option with strike price  𝐾=90 , and
# therefore the payoff is only computed at maturity:
def binomial_tree_call(S_ini, K, T, u, d, N):
    C = np.zeros([N + 1, N + 1])  # Call prices
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        C[N, i] = max(S_ini * (u ** (i)) * (d ** (N - i)) - K, 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return S, C

def binomial_tree_put(S_ini, K, T, u, d, N):
    P = np.zeros([N + 1, N + 1])  # Call prices
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        P[N, i] = max(K - S_ini * (u ** (i)) * (d ** (N - i)), 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return S, P

# Stock, Call = binomial_tree_put(50, 20, 2, 1.2, 0.8, 2)
# print("Underlying Price Evolution:\n", Stock)
# print("Call Option Payoff:\n", Call)

# ## 3. Introducing Risk-Neutral Probabilities and backward induction of Call Option Value
def binomial_call_full(S_ini, K, T, r, u, d, N):
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # Risk neutral probabilities (probs)
    C = np.zeros([N + 1, N + 1])  # Call prices
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        C[N, i] = max(S_ini * (u ** (i)) * (d ** (N - i)) - K, 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return C[0, 0], C, S

def binomial_put_full(S_ini, K, T, r, u, d, N):
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # Risk neutral probabilities (probs)
    Put = np.zeros([N + 1, N + 1])  # Put prices
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        Put[N, i] = max(K - S_ini * (u ** (i)) * (d ** (N - i)), 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            Put[j, i] = np.exp(-r * dt) * (p * Put[j + 1, i + 1] + (1 - p) * Put[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return Put[0, 0], Put, S

# call_price, C, S = binomial_call_full(100, 60, 1, 0, 1.2, 0.8, 1)
# print("Underlying Price Evolution:\n", S)
# print("Call Option Payoff:\n", C)
# print("Call Option Price at t=0: ", "{:.2f}".format(call_price))

# Binomial Tree for Put Options
def binomial_call_full_new(S_ini, K, T, r, u, d, N):
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # Risk neutral probs
    C = np.zeros([N + 1, N + 1])  # Call prices
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        C[N, i] = max(S_ini * (u ** (i)) * (d ** (N - i)) - K, 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return C[0, 0], C, S

def binomial_put_full_new(S_ini, K, T, r, u, d, N):
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # Risk neutral probs
    P = np.zeros([N + 1, N + 1])  # Call prices
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        P[N, i] = max(K - (S_ini * (u ** (i)) * (d ** (N - i))), 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            P[j, i] = np.exp(-r * dt) * (p * P[j + 1, i + 1] + (1 - p) * P[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return P[0, 0], P, S

# put_price, P, S = binomial_put_full_new(100, 98, 2, 0, 1.1, 0.8, 2)
# print("Underlying Price Evolution:\n", S)
# print("Put Option Payoff:\n", P)
# print("Price at t=0 for Put option with K=90 is $", "{:.2f}".format(put_price))

# Put-Call Parity in the Binomial Tree
# put_price, P, S = binomial_put_full(100, 90, 10, 0, 1.2, 0.8, 10)
# print("Price at t=0 for Put option with K=90 is $", "{:.2f}".format(put_price))
# call_price, C, S = binomial_call_full(100, 90, 10, 0, 1.2, 0.8, 10)
# print("Price at t=0 for Call option with K=90 is $", "{:.2f}".format(call_price))
# print(call_price + 90 * np.exp(-0 * 1) == S[0, 0] + put_price)

# Delta Hedging in the Binomial Tree
def call_option_delta(S_ini, K, T, r, u, d, N):
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probs
    C = np.zeros([N + 1, N + 1])  # call prices
    S = np.zeros([N + 1, N + 1])  # underlying price
    Delta = np.zeros([N, N])  # delta
    for i in range(0, N + 1):
        C[N, i] = max(S_ini * (u ** (i)) * (d ** (N - i)) - K, 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
            Delta[j, i] = (C[j + 1, i + 1] - C[j + 1, i]) / (
                S[j + 1, i + 1] - S[j + 1, i]
            )
    return C[0, 0], C, S, Delta

def put_option_delta(S_ini, K, T, r, u, d, N):
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probs
    Put = np.zeros([N + 1, N + 1])  # put prices
    S = np.zeros([N + 1, N + 1])  # underlying price
    Delta = np.zeros([N, N])  # delta
    for i in range(0, N + 1):
        Put[N, i] = max(K - S_ini * (u ** (i)) * (d ** (N - i)), 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            Put[j, i] = np.exp(-r * dt) * (p * Put[j + 1, i + 1] + (1 - p) * Put[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
            Delta[j, i] = (Put[j + 1, i + 1] - Put[j + 1, i]) / (
                S[j + 1, i + 1] - S[j + 1, i]
            )
    return Put[0, 0], Put, S, Delta

# price, call, S, delta = put_option_delta(100, 90, 1, 0, 1.1, 0.8, 1)
# print("Underlying: \n", S)
# print("Call Price: \n", call)
# print("Delta: \n", delta)

def call_option_delta_new(S_ini, K, T, r, sigma, N):
    dt = T / N  # Define time step
    u = np.exp(sigma * np.sqrt(dt))  # Define u
    d = np.exp(-sigma * np.sqrt(dt))  # Define d
    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probs
    C = np.zeros([N + 1, N + 1])  # call prices
    S = np.zeros([N + 1, N + 1])  # underlying price
    Delta = np.zeros([N, N])  # delta
    for i in range(0, N + 1):
        C[N, i] = max(S_ini * (u ** (i)) * (d ** (N - i)) - K, 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
            Delta[j, i] = (C[j + 1, i + 1] - C[j + 1, i]) / (
                S[j + 1, i + 1] - S[j + 1, i]
            )
    return C[0, 0], C, S, Delta

def put_option_delta_new(S_ini, K, T, r, sigma, N):
    dt = T / N  # Define time step
    u = np.exp(sigma * np.sqrt(dt))  # Define u
    d = np.exp(-sigma * np.sqrt(dt))  # Define d
    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probs

    print("sigma is ",sigma)
    print("r is ",r)
    print("dt is ",dt)
    print("p is ",p)
    print("q is ", 1 - p)

    Put = np.zeros([N + 1, N + 1])  # call prices
    S = np.zeros([N + 1, N + 1])  # underlying price
    Delta = np.zeros([N, N])  # delta
    for i in range(0, N + 1):
        Put[N, i] = max(K - S_ini * (u ** (i)) * (d ** (N - i)), 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            Put[j, i] = np.exp(-r * dt) * (p * Put[j + 1, i + 1] + (1 - p) * Put[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
            Delta[j, i] = (Put[j + 1, i + 1] - Put[j + 1, i]) / (
                S[j + 1, i + 1] - S[j + 1, i]
            )
    return Put[0, 0], Put, S, Delta

# def binomial_tree(S_ini, T, u, d, N)
# Stock = binomial_tree(100, 1, 1.2, 0.8, 2)
# print(Stock)

# def binomial_tree_call(S_ini, K, T, u, d, N):
# def binomial_tree_put(S_ini, K, T, u, d, N):

# Stock, Call = binomial_tree_call(100,90,1,1.1,0.8,1)
# print("Underlying Price Evolution:\n", Stock)
# print("Call Option Payoff:\n", Call)


# def binomial_put_full(S_ini, K, T, r, u, d, N):
# def binomial_call_full(S_ini, K, T, r, u, d, N):
# def binomial_put_full_new(S_ini, K, T, r, u, d, N):
# call_price, C, S = binomial_put_full_new(36,31,50,0.01,1.2,0.8,50)
# print("Underlying Price Evolution:\n", S)
# print("Call Option Payoff:\n", C)
# print("Call Option Price at t=0: ", "{:.2f}".format(call_price))


# put_price, C, S = binomial_call_full(100,90,1,0,1.1,0.8,1)
# print("Underlying Price Evolution:\n", S)
# print("Call Option Payoff:\n", C)
# print("Call Option Price at t=0: ", "{:.2f}".format(put_price))

# def call_option_delta(S_ini, K, T, r, u, d, N):
price, call, S, delta = call_option_delta(100,90,1,0,1.1,0.8,1)
print("Underlying: \n", S)
print("Call Price: \n", call)
print("Delta: \n", delta)


# call_option_delta_new(S_ini, K, T, r, sigma, N)
# Callvalue, Call, underlying_price, Delta = call_option_delta_new(100,80,1,)
# print(Delta)


# put_option_delta_new(100, 90, 1, 0.1,0.3, 10)