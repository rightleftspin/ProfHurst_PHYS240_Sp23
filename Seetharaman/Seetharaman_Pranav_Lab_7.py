import sympy as sy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmfit


def general_fit_function(function_string, x_data, y_data, guess_parameters):
    """
    This function takes in a function string to be parsed by Sympy's
    expression parser and tries to fit the data given along the 'x'
    independent variables. 
    """
    # Use sympy's parser to find the expression from the input string
    parsed_expression = sy.parsing.sympy_parser.parse_expr(function_string)
    parsed_function = sy.lambdify(list(parsed_expression.free_symbols), parsed_expression)
    
    # Create a model based on this input string with the independent variable x
    parsed_lm_model = lmfit.Model(parsed_function, independent_vars = ['x'])
    # Run the fit of the model to the data starting with the given parameters
    fit_to_model = parsed_lm_model.fit(data = y_data, x = x_data, **guess_parameters)

    return(fit_to_model, parsed_function)

def fit_to_parabola(x_data, y_data, guess_parameters):
    """
    Takes a set of x_data, y_data and initial parameters for a guess
    and fits this all to a parabola with no constant offset
    """

    parabola_string = "a * x + b * x ** 2"
    parabola_fit, parabola_function = general_fit_function(parabola_string, x_data, y_data, guess_parameters)

    return(parabola_fit, parabola_function)

# Define interesting initial velocities and initial guess parameters
initial_vels = [10, 20, 30, 40, 50, 500]
guess_param = {'a':1 , 'b':0 }

for velocity in initial_vels:
    # read data from balle output
    data = pd.read_pickle(f"./balle_data/data_{velocity}v.pickle")
    # transform data to numpy arrays
    x_data, y_data = data.iloc[[0]].to_numpy(), data.iloc[[1]].to_numpy()
    # fit the data to a parabolic model and take the parabolic function out
    fit_model, parabola_function = fit_to_parabola(x_data, y_data, guess_param)

    # do the same for the data with no air resistance
    data_noAir = pd.read_pickle(f"./balle_data/data_{velocity}v_na.pickle")
    x_data_noAir, y_data_noAir = data_noAir.iloc[[0]].to_numpy(), data_noAir.iloc[[1]].to_numpy()
    fit_model_noAir, _ = fit_to_parabola(x_data_noAir, y_data_noAir, guess_param)

    # find the chi-suqared values for each
    print(f"Chi-Squared for {velocity} m/s with air resistance is {fit_model.chisqr:.3e}")
    print(f"Chi-Squared for {velocity} m/s without air resistance is {fit_model_noAir.chisqr:.3e}")

    plt.xlim([0, x_data_noAir.max() * 1.2])
    plt.ylim([0, y_data_noAir.max() * 1.2])
        
    # Find the fit y_data by passing the initial x_data through the parabolic model
    y_data_fit = parabola_function(x = x_data, **fit_model.values)
    # same for no air resistance
    y_data_fit_noAir = parabola_function(x = x_data_noAir, **fit_model_noAir.values)
    
    # plot the data alongside the model itself for visual comparison
    plt.plot(x_data.transpose(), y_data_fit[0], 'k-', label = "Air Resistance Best Fit")
    plt.plot(x_data, y_data, 'k.', label = "Air Resistance")

    # smae for no air resistance
    plt.plot(x_data_noAir.transpose(), y_data_fit_noAir[0], 'b-', label = "No Air Resistance Best Fit" )
    plt.plot(x_data_noAir, y_data_noAir, 'b.', label = "No Air Resistance")

    plt.xlabel("Range (m)")
    plt.ylabel("Height (m)")
    plt.suptitle("Trajectory of a ball with and without Air Resistance")
    plt.title(f"Initial Velocity {velocity}m/s")

    # apparently matplotlib defines an individual label for each point 
    # in the legend, so this just removes redundancy in the legend
    # by forcing it into a dictionary so only unique values survive
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(f"model_for_{velocity}v.pdf")
    plt.close()
    

