import numpy as np

class BinomialTreeOption:
    def __init__(self, S0, E, T, r, sigma, n, option_type="call"):
        """
        :param S0: Initial stock price
        :param E: Strike price
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate
        :param sigma: Volatility of the stock
        :param n: Number of steps in the binomial tree
        :param option_type: "call" for call option, "put" for put option
        """
        self.S0 = S0
        self.E = E
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n = n
        self.option_type = option_type.lower()

    def price(self):
        dt = self.T / self.n  # Time step
        u = np.exp(self.sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.r * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize asset prices at maturity
        stock_prices = np.zeros(self.n + 1)
        option_values = np.zeros(self.n + 1)

        for i in range(self.n + 1):
            stock_prices[i] = self.S0 * (u ** (self.n - i)) * (d ** i)

        # Calculate option values at maturity
        if self.option_type == "call":
            option_values = np.maximum(stock_prices - self.E, 0)
        elif self.option_type == "put":
            option_values = np.maximum(self.E - stock_prices, 0)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        # Step back through the tree to get present value
        for j in range(self.n - 1, -1, -1):
            for i in range(j + 1):
                stock_prices[i] = self.S0 * (u ** (j - i)) * (d ** i)
                option_values[i] = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])

                # Early exercise condition for American options
                if self.option_type == "call":
                    option_values[i] = max(option_values[i], stock_prices[i] - self.E)
                elif self.option_type == "put":
                    option_values[i] = max(option_values[i], self.E - stock_prices[i])

        return round(option_values[0], 2)

# Example usage
S0 = 95  # Current stock price
E = 100  # Strike price
T = 1  # Time to expiration (in years)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
n = 100  # Number of binomial steps

# Call option pricing
call_option = BinomialTreeOption(S0, E, T, r, sigma, n, "call")
print(f"American Call Option Price: ${call_option.price()}")

# Put option pricing
put_option = BinomialTreeOption(S0, E, T, r, sigma, n, "put")
print(f"American Put Option Price: ${put_option.price()}")
