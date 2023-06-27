import pygad
import numpy as np
from flask import Flask, jsonify, render_template, request

# Define the fitness function


def evaluate_fitness(solution, sol_idx):
    # Add your fitness evaluation logic here
    fitness_value = ...
    return fitness_value

# Define the main function to run the Genetic Algorithm


def run_genetic_algorithm():
    # Define the function to calculate fitness
    def fitness_function(solution, sol_idx): return evaluate_fitness(
        solution, sol_idx)

    # Create an instance of the pyGAD.GA class
    ga_instance = pygad.GA(num_generations=10,
                           num_parents_mating=10,
                           fitness_func=fitness_function)

    # Run the GA
    ga_instance.run()

    # Retrieve the best solution
    best_solution = ga_instance.best_solution()
    best_fitness = ga_instance.best_solution()[1]

    # Return the results
    return best_solution, best_fitness

# Flask backend


ENV = 'dev'
app = Flask(__name__)

if ENV == 'dev':
    app.debug = True
else:
    app.debug = False


@app.route('/', methods=['GET'])
def index():
    population = 10  # best 10
    genes = 5  # or features or k
    rgb = 3
    colour = np.random.randint(0, 255, size=(genes, rgb))
    # ratio = np.random.randn((population,genes))
    return render_template('index.html', data={'colour': colour})


@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')


# @app.route('/evolve', methods=['POST'])
@app.route('/evolve', methods=['GET'])
def evolve():
    formData = request.values
    # fitness = request.form['fitness']  # {}
    for key in formData:
        value = formData.get(key)
        if key in "submit":
            print(key)
        print(key, value)
    # if request.method == 'POST':
    #     fitness = request.form['fitness']
    return jsonify(formData)
# Route to start the GA


@app.route('/start_ga', methods=['GET'])
def start_genetic_algorithm():
    # best_solution, best_fitness = run_genetic_algorithm()
    best_solution, best_fitness = 0.2, 1
    response = {
        'best_solution': best_solution,
        'best_fitness': best_fitness
    }
    return jsonify(response)


# Run the Flask app
if __name__ == '__main__':
    app.run()
