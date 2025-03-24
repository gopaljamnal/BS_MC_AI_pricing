from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)


class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_simulation(self):
        # we have 2 columns: first with 0s the second column will store the payoff
        # we need the first column of 0s: payoff function is max(0,S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        # dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        # we need S-E because we have to calculate the max(S-E,0)
        option_data[:, 1] = stock_price - self.E

        # average for the Monte-Carlo simulation
        # max() returns the max(0,S-E) according to the formula
        # THIS IS THE AVERAGE VALUE !!!
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # have to use the exp(-rT) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average

    def put_option_simulation(self):
        # we have 2 columns: first with 0s the second column will store the payoff
        # we need the first column of 0s: payoff function is max(0,S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        # dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2)
                                       + self.sigma * np.sqrt(self.T) * rand)

        # we need S-E because we have to calculate the max(E-S,0)
        option_data[:, 1] = self.E - stock_price

        # average for the Monte-Carlo simulation
        # max() returns the max(0,S-E) according to the formula
        # THIS IS THE AVERAGE VALUE !!!
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # have to use the exp(-rT) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        # Get parameters from the form
        S0 = float(request.form.get('S0'))
        E = float(request.form.get('E'))
        T = float(request.form.get('T'))
        rf = float(request.form.get('rf'))
        sigma = float(request.form.get('sigma'))
        iterations = int(request.form.get('iterations'))

        # Create pricing model using the provided class
        model = OptionPricing(S0, E, T, rf, sigma, iterations)

        # Calculate option prices
        call_price = model.call_option_simulation()
        put_price = model.put_option_simulation()

        return jsonify({
            'success': True,
            'call_price': round(call_price, 4),
            'put_price': round(put_price, 4)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True)
